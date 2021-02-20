#include "flatbuffers/flatbuffers.h"
#include "schema_generated.h"
#include <iostream>
#include <fstream>

//const int TRANSPOSE_OP_CODE = 39;
//const int CONV_2D_OP_CODE = 3;

// Note this is the index into the operator codes, but too lazy.
const int PAD_OP_CODE_IDX = 1;
const int TRANSPOSE_OP_CODE_IDX = 2;
const int CONV_2D_OP_CODE_IDX = 3;
const int ADD_OP_CODE_IDX = 4;
const int MEAN_OP_CODE_IDX = 5;
const int GATHER_OP_CODE_IDX = 6;
const int SUB_OP_CODE_IDX = 7;


int main() {
  // This is ctrl-c from the tutorial
  std::ifstream infile;
  infile.open("model.tflite", std::ios::binary | std::ios::in);
  infile.seekg(0,std::ios::end);
  int length = infile.tellg();
  infile.seekg(0,std::ios::beg);
  char *data = new char[length];
  infile.read(data, length);
  infile.close();
  // end
 
  tflite::ModelT model;
  tflite::GetModel(data)->UnPackTo(&model);

  auto& subgraph = model.subgraphs.at(0);
  auto& tensors = subgraph->tensors;
  auto& ops = subgraph->operators;
  
  auto& reshape_op = ops.at(0);
  int reshape_dim_tensor_idx = reshape_op->inputs.at(1);
  auto& reshape_tensor = tensors.at(reshape_dim_tensor_idx);
  int buffer_id = reshape_tensor->buffer;
  auto& buffers = model.buffers;
  auto& reshape_buffer = buffers.at(buffer_id);
  auto& reshape_data = reshape_buffer->data;
  // little-endian
  uint8_t new_index = 3;
  reshape_data[0] = new_index;
  
  // shapes are static too, so need to change this
  int output_tensor_idx = reshape_op->outputs.at(0);
  auto& shape = tensors.at(output_tensor_idx)->shape;
  // BCHW -> BHWC
  int C = shape[1];
  shape[1] = shape[2];
  shape[2] = shape[3];
  shape[3] = C;

  // and the pad...
  /*{
  auto& pad_op = ops.at(1);
  int output_tensor_idx = pad_op->outputs.at(1);
  auto& shape = tensors.at(output_tensor_idx)->shape;
  shape[0] = 1;
  shape[1] = 38;
  shape[2] = 1;
  shape[3] = 1;
  }
  */

  std::vector<int> ops_to_drop;
  // pad inputs are a const, so make sure we only modify it once
  std::set<int> pad_tensor_ids;

  // Re-wire the following cases:
  // 1. X -> TRANSPOSE -> CONV_2D into X -> CONV_2D
  // 2. CONV_2D -> TRANSPOSE -> X into CONV_2D -> X
  // 3. ADD -> TRANSPOSE -> X     into ADD -> X
  // 4. SUB -> TRANSPOSE ->X      into SUB -> X
  //
  // and transpose any pad output shape from BCHW into BHWC
  for (int i = 0; i < ops.size(); i++) {
    if (i == ops.size() - 3) {
      break;
    }
    auto& op = ops.at(i);
    auto& n_op = ops.at(i+1);
    auto& nn_op = ops.at(i+2);
    if (n_op->opcode_index == TRANSPOSE_OP_CODE_IDX && 
        nn_op->opcode_index == CONV_2D_OP_CODE_IDX) {
      // Case 1
      std::cout << "found x->transpose->conv2d" << std::endl;
      // alias things for readability
      auto& x = op;
      auto& transpose = n_op;
      auto& conv2d = nn_op;

      int x_out = x->outputs.at(0);
      conv2d->inputs[0] = x_out;

      transpose->inputs.clear();
      transpose->outputs.clear();
      ops_to_drop.push_back(i+1);
    } else if (op->opcode_index == CONV_2D_OP_CODE_IDX 
        && n_op->opcode_index == TRANSPOSE_OP_CODE_IDX) {
      // Case 2
      std::cout << "found conv2d->transpose->x" << std::endl;
      // alias things for readability
      auto& conv2d = op;
      auto& transpose = n_op;
      auto& x = nn_op;

      int conv2d_out = conv2d->outputs.at(0);
      x->inputs.at(0) = conv2d_out;

      // Safe to delete the tranpose op now.
      transpose->inputs.clear();
      transpose->outputs.clear();
      ops_to_drop.push_back(i+1);
    } else if (op->opcode_index == ADD_OP_CODE_IDX 
        && n_op->opcode_index == TRANSPOSE_OP_CODE_IDX) {
      // Case 3
      std::cout << "found add->transpose->x" << std::endl;
      // alias things for readability
      auto& add = op;
      auto& transpose = n_op;
      auto& x = nn_op;

      int add_out = add->outputs.at(0);
      x->inputs.at(0) = add_out;

      // Safe to delete the tranpose op now.
      transpose->inputs.clear();
      transpose->outputs.clear();
      ops_to_drop.push_back(i+1);
    } else if (op->opcode_index == SUB_OP_CODE_IDX && n_op->opcode_index == TRANSPOSE_OP_CODE_IDX) {
      // Case 4
      auto& sub = op;
      auto& transpose = n_op;
      auto& x = nn_op;

      int sub_out = op->outputs.at(0);
      x->inputs.at(0) = sub_out;

      auto& sub_out_tensor = tensors.at(sub_out);
      auto& shape = sub_out_tensor->shape;
      int C = shape[1];
      shape[1] = shape[2];
      shape[2] = C;

      // Safe to delete the tranpose op now.
      transpose->inputs.clear();
      transpose->outputs.clear();
      ops_to_drop.push_back(i+1);

      // Hack to rewire the output of the graph
      subgraph->outputs.at(1) = sub_out;
    }


    // reshape the pad operations
    if (op->opcode_index == PAD_OP_CODE_IDX) {
      std::cout << "reshaping pad from BCHW to BHWC" << std::endl;
      auto& pad_op = op;
      int output_tensor_idx = pad_op->outputs.at(0);
      auto& shape = tensors.at(output_tensor_idx)->shape;
      int C = shape[1];
      shape[1] = shape[2];
      shape[2] = shape[3];
      shape[3] = C;

      // now we also need to re-arrange the tensor
      auto& in_tensor_idx = pad_op->inputs.at(1);
      if (pad_tensor_ids.count(in_tensor_idx) > 0) {
        continue;
      } else {
        pad_tensor_ids.insert(in_tensor_idx);
      }
      std::cout << "tensor_id: " << in_tensor_idx << std::endl;
      auto& in_tensor = tensors.at(in_tensor_idx);
      auto& in_shape = in_tensor->shape;
      int buffer_id = in_tensor->buffer;
      std::cout << "buffer_id: " << buffer_id << std::endl;
      auto& pad_buffer = buffers.at(buffer_id);
      auto& pad_data = pad_buffer->data;
      // pad is int32 -> 4 bytes
      // pad_data size is 4 * rows * cols (4 * 4 * 2) == 32
      // pad_data is stored in row-major order
      // pad_data[i][j] is at pad_data[(i * (cols * 4) + (j * 4)]
      int rows = in_shape.at(0);
      int cols = in_shape.at(1);

      // Pad is currently BCHW, put we want to pad for BHWC
      // We want pad_data[2][:] and pad_data[3][:] and put it into
      // pad_data[1][:] and pad_data[2][:], respectively
      int i = 2;
      int j = 0;
      int new_i = 1;
      int new_j = 0;
      int old_idx;
      int new_idx;
      uint8_t zero = 0;
      for (int b = 0; b < 4; b++) {
        old_idx = i * (cols*4) + (j*4) + b;
        new_idx = new_i * (cols*4) + (new_j*4) + b;
        pad_data[new_idx] = pad_data[old_idx];
        pad_data[old_idx] = zero;
      }
      i = 2;
      j = 1;
      new_i = 1;
      new_j = 1;
      for (int b = 0; b < 4; b++) {
        old_idx = i * (cols*4) + (j*4) + b;
        new_idx = new_i * (cols*4) + (new_j*4) + b;
        pad_data[new_idx] = pad_data[old_idx];
        pad_data[old_idx] = zero;
      }
      i = 3;
      j = 0;
      new_i = 2;
      new_j = 0;
      for (int b = 0; b < 4; b++) {
        old_idx = i * (cols*4) + (j*4) + b;
        new_idx = new_i * (cols*4) + (new_j*4) + b;
        pad_data[new_idx] = pad_data[old_idx];
        pad_data[old_idx] = zero;
      }
      i = 3;
      j = 1;
      new_i = 2;
      new_j = 1;
      for (int b = 0; b < 4; b++) {
        old_idx = i * (cols*4) + (j*4) + b;
        new_idx = new_i * (cols*4) + (new_j*4) + b;
        pad_data[new_idx] = pad_data[old_idx];
        pad_data[old_idx] = zero;
      }
    } else if (op->opcode_index == MEAN_OP_CODE_IDX) {
      int mean_tensor_idx = op->inputs.at(1);
      // TODO: Hack for this graph
      if (mean_tensor_idx != 25) {
        continue;
      }

      // shape
      int mean_out_tensor_idx = op->outputs.at(0);
      auto& mean_out_tensor = tensors.at(mean_out_tensor_idx);
      auto& shape = mean_out_tensor->shape;
      int C = shape[1];
      shape[1] = shape[2];
      shape[2] = shape[3];
      shape[3] = C;


      auto& mean_tensor = tensors.at(mean_tensor_idx);
      int buffer_id = mean_tensor->buffer;
      auto& mean_buffer = buffers.at(buffer_id);
      auto& mean_data = mean_buffer->data;
      // little-endian
      uint8_t new_index = 1;
      mean_data[0] = new_index;
    } else if (op->opcode_index == GATHER_OP_CODE_IDX) {
      tflite::GatherOptionsT gather_options;
      //tflite::GetModel(data)->UnPackTo(&model);
      auto opts = op->builtin_options.AsGatherOptions();
      opts->axis = 1;

      // shape
      int gather_out_tensor_idx = op->outputs.at(0);
      auto& gather_out_tensor = tensors.at(gather_out_tensor_idx);
      auto& shape = gather_out_tensor->shape;
      int C = shape[1];
      shape[1] = shape[2];
      shape[2] = C;
    } 
  } // end for 

  auto end = ops.end();
  int ops_size = ops.size();
  for (int i = ops_to_drop.size() - 1; i >= 0; i--) {
    //std::cout << ops_to_drop.at(i) << std::endl;
    int from_end = ops_size - ops_to_drop.at(i);
    //std::cout<< "deleting: " << from_end << std::endl;
    ops.erase(end - from_end);
  }

  // finish
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(tflite::Model::Pack(fbb, &model), tflite::ModelIdentifier());

  // Write
  uint8_t *buf = fbb.GetBufferPointer();
  int size = fbb.GetSize();
  std::ofstream ofile("model-mutated.tflite", std::ios::binary);
  ofile.write((char *)buf, size);
  ofile.close();

  return 0;
}

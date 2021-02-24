Tensorflow Lite headers generated with:
```
flatc -p -b -c --gen-object-api --gen-mutable ~/src/tensorflow/tensorflow/lite/schema/schema.fbs
```

Compile with (you need to flatbuffers headers somewhere too):
```
clang++ -std=c++11 -I${HOME}/src/flatbuffers/include edit_tflite.cc
```

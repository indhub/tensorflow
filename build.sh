bazel build --copt=-O1 -c dbg -c opt --linkopt=-rdynamic --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

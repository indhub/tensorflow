bazel build --linkopt=-rdynamic --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

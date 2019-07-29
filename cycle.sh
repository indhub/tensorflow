set -e
bash build.sh
bash package.sh
pip uninstall --yes tensorflow
pip install /tmp/tensorflow_pkg/*.whl

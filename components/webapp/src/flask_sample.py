'''
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import logging
import os
# from threading import Timer

from flask import Flask, render_template, request


app = Flask(__name__)
class Score:
    def __init__(self):
        self.val=10


# handle requests to the server
@app.route("/")
def main():

  connection = {"text": "", "success": False}
  output = {"truth": "truth", "prediction": "pred",
            "img_path": "serving_path", "scores": [Score()]}
  args = {"name": "name_arg", "addr": "addr_arg", "port": "port_arg", "version": str("model_version")}
 
  return render_template('index.html', output=output,
                         connection=connection, args=args)


def remove_resource(path):
  """
  attempt to delete file from path. Used to clean up MNIST testing images

  :param path: the path of the file to delete
  """
  try:
    os.remove(path)
    print("removed " + path)
  except OSError:
    print("no file at " + path)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO,
                      format=('%(levelname)s|%(asctime)s'
                              '|%(pathname)s|%(lineno)d| %(message)s'),
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      )
  logging.getLogger().setLevel(logging.INFO)
  logging.info("Starting flask.")
  app.run(debug=True, host='0.0.0.0', port=5000)

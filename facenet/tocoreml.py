import tfcoreml as tf_converter

tf_converter.convert(tf_model_path = 'facenet.pb',
                     mlmodel_path = 'facenet.mlmodel',
                     image_input_names='input:0',
                     output_feature_names = ['output:0'])
import tfcoreml as tf_converter

tf_converter.convert(tf_model_path = 'frozen_graphs/pnet.pb',
                     mlmodel_path = 'pnet.mlmodel',
                     image_input_names='input:0',
                     output_feature_names = ["cls_prob:0","bbox_pred:0","landmark_pred:0"])

tf_converter.convert(tf_model_path = 'frozen_graphs/rnet.pb',
                     mlmodel_path = 'rnet.mlmodel',
                     image_input_names='input:0',
                     output_feature_names = ["cls_prob:0","bbox_pred:0","landmark_pred:0"])


tf_converter.convert(tf_model_path = 'frozen_graphs/onet.pb',
                     mlmodel_path = 'onet.mlmodel',
                     image_input_names='input:0',
                     output_feature_names = ["cls_prob:0","bbox_pred:0","landmark_pred:0"])
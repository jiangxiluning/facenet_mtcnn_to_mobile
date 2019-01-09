#!/bin/bash
tflite_convert --output_file MTCNN_mobile/pnet.tflite --graph_def_file frozen_graphs/pnet.pb  --input_arrays "input" --input_shapes "1,600,800,3" --output_arrays cls_prob,bbox_pred,landmark_pred --output_format TFLITE
echo "PNet is converted successfully."
tflite_convert --output_file MTCNN_mobile/rnet.tflite --graph_def_file frozen_graphs/rnet.pb  --input_arrays "input" --input_shapes "64,24,24,3" --output_arrays cls_prob,bbox_pred,landmark_pred --output_format TFLITE
echo "RNet is converted successfully."
tflite_convert --output_file MTCNN_mobile/onet.tflite --graph_def_file frozen_graphs/onet.pb  --input_arrays "input" --input_shapes "16,48,48,3" --output_arrays cls_prob,bbox_pred,landmark_pred --output_format TFLITE
echo "ONet is converted successfully."
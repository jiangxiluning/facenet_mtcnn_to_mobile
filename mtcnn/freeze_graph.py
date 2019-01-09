import tensorflow as tf
from src.mtcnn_model import P_Net, R_Net, O_Net
import click
from pathlib import Path

MODEL_DIR = (Path(__file__).parent / 'MTCNN_model').absolute()
OUTPUT_DIR = (Path(__file__).parent / "frozen_graphs").absolute()

def freeze_graph_def(sess, input_graph_def, output_node_names):
    # Replace all the variables in the graph with constants of the same values
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","))
    return output_graph_def

def save_pnet(width, height):
    model_path = MODEL_DIR / "PNet_landmark/PNet-18"
    graph = tf.Graph()
    with graph.as_default():
        #define tensor and op in graph(-1,1)
        image_op = tf.placeholder(tf.float32, shape=(1, height, width, 3), name='input')
        cls_prob, bbox_pred, _ = P_Net(image_op, training=False)
        
        #allow 
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, model_path.as_posix())

        # Retrieve the protobuf graph definition and fix the batch norm nodes
        input_graph_def = sess.graph.as_graph_def()
            
        # Freeze the graph def
        output_graph_def = freeze_graph_def(sess, input_graph_def, 'cls_prob,bbox_pred,landmark_pred')
    
        output_pnet = OUTPUT_DIR / 'pnet.pb'
        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_pnet.as_posix(), 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_pnet))
    

def save_onet():
    model_path = MODEL_DIR / "ONet_landmark/ONet-16"
    graph = tf.Graph()
    with graph.as_default():
            image_op = tf.placeholder(tf.float32, shape=[16, 48, 48, 3], name='input')
            #figure out landmark            
            cls_prob, bbox_pred, landmark_pred = O_Net(image_op, training=False)
            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, model_path.as_posix())

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
                
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'cls_prob,bbox_pred,landmark_pred')
        
            output_pnet = OUTPUT_DIR / 'onet.pb'
            # Serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_pnet.as_posix(), 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_pnet))


def save_rnet():
    model_path = MODEL_DIR / "RNet_landmark/RNet-14"
    graph = tf.Graph()
    with graph.as_default():
            image_op = tf.placeholder(tf.float32, shape=[64, 24, 24, 3], name='input')
            #figure out landmark            
            cls_prob, bbox_pred, landmark_pred = R_Net(image_op, training=False)
            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, model_path.as_posix())

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
                
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'cls_prob,bbox_pred,landmark_pred')
        
            output_pnet = OUTPUT_DIR / 'rnet.pb'
            # Serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_pnet.as_posix(), 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_pnet))


@click.command()
@click.option('--width', default=800)
@click.option('--height', default=600)
def main(width, height):
    save_pnet(width, height)
    save_rnet()
    save_onet()

if __name__ == "__main__":
    main()
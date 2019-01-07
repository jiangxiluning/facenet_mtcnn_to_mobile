import tensorflow as tf
from src.models import inception_resnet_v1
import sys
import click
from pathlib import Path


@click.command()
@click.argument('training_checkpoint_dir', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument('eval_checkpoint_dir', type=click.Path(exists=True, file_okay=False, resolve_path=True))
def main(training_checkpoint_dir, eval_checkpoint_dir):

    traning_checkpoint = Path(training_checkpoint_dir) / "model-20180402-114759.ckpt-275"
    eval_checkpoint = Path(eval_checkpoint_dir) / "imagenet_facenet.ckpt"

    data_input = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 160, 160, 3])

    output, _ = inception_resnet_v1.inference(data_input, keep_probability=0.8, phase_train=False, bottleneck_layer_size=512)
    output = tf.identity(output, name='output')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, traning_checkpoint.as_posix())
        save_path = saver.save(sess, eval_checkpoint.as_posix())
        print("Model saved in file: %s" % save_path)




if __name__ == "__main__":
    main()
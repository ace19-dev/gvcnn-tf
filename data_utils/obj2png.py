"""
Created on Sat Jul  7 00:40:00 2018

@author: Peter M. Clausen, pclausen

MIT License

Copyright (c) 2018 pclausen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import tensorflow as tf

import os
import data_utils.ObjFile
import sys
import os
import glob


flags = tf.app.flags

FLAGS = flags.FLAGS


flags.DEFINE_string('source_dir', '/home/ace19/dl_data/ModelNet10',
                    'Directory where .obj files is.')

flags.DEFINE_string('target_dir', '/home/ace19/dl_data/modelnet',
                    'Output directory.')

flags.DEFINE_string('target_file_ext', '.obj',
                    'target file extension')

flags.DEFINE_string('dataset_category', 'train',
                    'train or test')

flags.DEFINE_integer('num_views', 8, 'Number of views')
flags.DEFINE_float('azim', 45, 'Azimuth angle of view in degrees.')
flags.DEFINE_float('elevation', None, 'Elevation angle of view in degrees.')
flags.DEFINE_string('quality', 'LOW', 'Image quality (HIGH,MEDIUM,LOW).  Default: LOW')
flags.DEFINE_float('scale', 0.9,
                   'Scale picture by descreasing boundaries. Lower than 1. gives a larger object.')
flags.DEFINE_string('animate', None,
                    'Animate instead of creating picture file as animation, from elevation -180:180 and azim -180:180')

# parser.add_argument("-v", "--view",
#           dest='view',
#           action='store_true',
#           help="View instead of creating picture file.")


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    res={'HIGH':1200,'MEDIUM':600,'LOW':300}
    dpi=None
    if FLAGS.quality:
        if type(FLAGS.quality)==int:
            dpi=FLAGS.quality
        elif FLAGS.quality.upper() in res:
            dpi=res[FLAGS.quality.upper()]

    azim=None
    if FLAGS.azim is not None:
        azim=FLAGS.azim

    elevation=None
    if FLAGS.elevation is not None:
        elevation=FLAGS.elevation

    scale=None
    if FLAGS.scale:
        scale=FLAGS.scale

    animate=None
    if FLAGS.animate:
        animate=FLAGS.animate


    root = os.listdir(FLAGS.source_dir)
    root.sort()
    for cls in root:
        if not os.path.isdir(os.path.join(FLAGS.source_dir, cls)):
            continue

        dataset = os.path.join(FLAGS.source_dir, cls, FLAGS.dataset_category)
        files = os.listdir(dataset)
        files.sort()

        for objfile in files:
            obj_file_path = os.path.join(dataset, objfile)
            if os.path.isfile(obj_file_path) and '.obj' in objfile:
                target_path = objfile.replace('.obj','.off')
                # if FLAGS.outfile:
                #     outfile=FLAGS.outfile
                # if FLAGS.view:
                #     outfile=None
                # else:
                #     print('Converting %s to %s'%(objfile, outfile))

                ob = data_utils.ObjFile.ObjFile(obj_file_path)

                for i in range(FLAGS.num_views):
                    new_output = objfile[:-4] + '.' + str(i) + '.png'
                    print('Converting %s to %s' % (objfile, new_output))
                    outfile_path = os.path.join(FLAGS.target_dir, cls, FLAGS.dataset_category,
                                                target_path, new_output)
                    ob.Plot(outfile_path,
                            elevation=elevation,
                            azim=azim*(i+1),
                            dpi=dpi,
                            scale=scale,
                            animate=animate)
            # else:
            #     print('File %s not found or not file type .obj'%objfile)
            #     sys.exit(1)
    
if __name__ == '__main__':
    tf.app.run()
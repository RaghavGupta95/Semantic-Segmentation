from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, IMAGES, configure_uploads
import caffe
import cPickle as pickle
from PIL import Image
import numpy as np


net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
app = Flask(__name__)
classes = []
photos = UploadSet('photos', default_dest=lambda app: app.instance_root)
app.config['STATIC'] = 'static'
app.config['UPLOADED_PHOTOS_DEST'] = 'img'
configure_uploads(app, photos)
class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
'diningtable', 'dog', 'horse', 'motorbike', 'person',
'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
def paletize(arr):
    out_img = np.zeros((arr.shape[0],arr.shape[1],3),dtype=np.uint8)
    palette = [[116,64,206],[118,215,74],
                [206,78,196],[206,206,73],[83,44,121],
                [119,216,146],[193,72,125],[123,202,196],
                [211,70,58],[121,118,208],
                [214,136,58],[98,138,174],
                [144,122,57],[205,165,206],
                [60,80,56],[192,128,128],[205,201,153],
                [69,45,71],[0,192,0],[192,130,121],
                [114,52,41]]
    for v in np.unique(arr):
        out_img[arr==v]=palette[v]
    return out_img

#I want a preview
# and i want a postview
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        #dont save this
        image_data= request.files['photo'].read()
        bfname = request.files['photo'].filename
        infname = 'static/'+bfname
        import io
        im = Image.open(io.BytesIO(image_data))
        #print request.files['photo'].keys()
        im.save(infname)
        #filename = photos.save(request.files['photo'])
        #Get the data
        #im = Image.open("img/"+filename)
        #im.save("static/" + filename )
        #print np.array(class_names)[np.unique(im)]
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        #process it
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))
        #put it in the network
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        #run the network
        net.forward()
        #get the result
        out = net.blobs['score'].data[0].argmax(axis=0)
        classes = np.array(class_names)[np.unique(out)]
        final = paletize(out)
        print final.shape

        outfname = 'out_'+ bfname
        j = Image.fromarray(final)
        #dont save this
        j.save('static/out_'+ bfname)
        blended = Image.blend('static/'+bfname,'static/out_'+ bfname,alpha = 0.5)
        blended.save('blended.jpg')
        print bfname
        return render_template('template_2.html',outfilename=outfname,infilename=bfname,outdata = classes)

    return render_template('upload.html')


if __name__ == '__main__':
	app.run(debug=True)

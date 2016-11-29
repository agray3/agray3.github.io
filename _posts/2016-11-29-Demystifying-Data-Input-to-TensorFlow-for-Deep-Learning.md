---
layout: post
title: "Demystifying Data Input to TensorFlow for Deep Learning"
date: 2016-11-29
---

<h2> Introduction </h2>

[TensorFlow](https://www.tensorflow.org/) is an incredibly powerful new framework for deep learning. The "MNIST For ML Beginners" and "Deep MNIST for Experts" [TensorFlow tutorials](https://www.tensorflow.org/versions/r0.11/tutorials/index.html) give an excellent introduction to the framework. This article acts as a follow-on tutorial which addresses the following issues:

1. The above tutorials use the MNIST dataset of hand written numbers, which pre-exists in TensorFlow TFRecord format and is loaded automatically. This can be a bit mysterious if you have no experience of data format manipulation in TensorFlow.
2. Since the MNIST dataset is fixed, there is little scope for experimentation through adjusting the images and network to get a feel for how to deal with particular aspects of real data.


Here,

1. You create your own images in a standard "png" format (that you can easily view), and you convert to TensorFlow TFRecord format. These are images of shapes created from python using the matplotlib module. 
2. You are free to explore by changing the way the images are created (contents, resolution, number of classes ...).


The aim is to help you get to the point where you are comfortable in
using TensorFlow with your own data, and also provide the opportunity
for you to experiment by creating different datasets and adjusting the
neural network accordingly.  This tutorial assumes you are using a
UNIX based system such as Linux or OSX.


<h2>Shape Sorting</h2>

![Shape Sorter]({{ site.url }}/files/shape_sorter.jpg)


If you can't find a toddler to sort your shapes for you, don't worry: help is here. You are going to learn how to create a virtual shape sorting algorithm.


<h3>Creating the shapes</h3>

You will create images of shapes using the [Matplotlib](http://matplotlib.org)  python module. If you don't already have this on your system then please see the [installation instructions here](http://matplotlib.org/users/installing.html). 

We are going to use python to create images of shapes with random positions and sizes: to keep things simple we are going to stick to 2 classes (squares and triangles), and to keep training time reasonable we are going to use low resolution of 32x32 (similar to the 28x28 of MNIST) - after the tutorial you can adjust these to your satisfaction. 

First, create a new directory to work in:
<pre>
mkdir shapesorter
cd shapesorter
</pre>

Now set up some directories to contain training and validation data, for each of our two classes (<i>squares</i> and <i>triangles</i>) 
<pre>
mkdir -p data/train/squares
mkdir -p data/train/triangles
mkdir -p data/validate/squares
mkdir -p data/validate/triangles
</pre>

The python script to automatically create a set of squares and triangles is below. This uses random numbers to vary position and size of these shapes. Please read through the comments in the script which describe the different stages.


<!-- HTML generated using hilite.me --><div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">matplotlib.path</span> <span style="color: #008800; font-weight: bold">as</span> <span style="color: #0e84b5; font-weight: bold">mpath</span>
<span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">matplotlib.patches</span> <span style="color: #008800; font-weight: bold">as</span> <span style="color: #0e84b5; font-weight: bold">mpatches</span>
<span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">matplotlib.pyplot</span> <span style="color: #008800; font-weight: bold">as</span> <span style="color: #0e84b5; font-weight: bold">plt</span>
<span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">random</span>
<span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">math</span>

<span style="color: #888888">#number of images we are going to create in each of the two classes</span>
nfigs<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">4000</span>

<span style="color: #888888"># Specify the size of the image. </span>
<span style="color: #888888"># E.g. size=32 will create images with 32x32 pixels.</span>
size<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">32</span>

<span style="color: #888888">#loop over classes</span>
<span style="color: #008800; font-weight: bold">for</span> clss <span style="color: #000000; font-weight: bold">in</span> [<span style="background-color: #fff0f0">&quot;squares&quot;</span>,<span style="background-color: #fff0f0">&quot;triangles&quot;</span>]:
    <span style="color: #008800; font-weight: bold">print</span> <span style="background-color: #fff0f0">&quot;generating images of &quot;</span><span style="color: #333333">+</span>clss<span style="color: #333333">+</span><span style="background-color: #fff0f0">&quot;:&quot;</span>
    
    <span style="color: #888888">#loop over number of images to generate</span>
    <span style="color: #008800; font-weight: bold">for</span> i <span style="color: #000000; font-weight: bold">in</span> <span style="color: #007020">range</span>(nfigs):

        <span style="color: #888888">#initialise a new figure </span>
        fig, ax <span style="color: #333333">=</span> plt<span style="color: #333333">.</span>subplots()

        <span style="color: #888888">#initialise a new path to be used to draw on the figure</span>
        Path <span style="color: #333333">=</span> mpath<span style="color: #333333">.</span>Path

        <span style="color: #888888">#set position and scale of each shape using random numbers</span>
        <span style="color: #888888">#the coefficients are used to just try and prevent too many shapes from </span>
        <span style="color: #888888">#spilling off the edge of the image</span>
        basex<span style="color: #333333">=</span><span style="color: #6600EE; font-weight: bold">0.7</span><span style="color: #333333">*</span>random<span style="color: #333333">.</span>random()
        basey<span style="color: #333333">=</span><span style="color: #6600EE; font-weight: bold">0.7</span><span style="color: #333333">*</span>random<span style="color: #333333">.</span>random()
        length<span style="color: #333333">=</span><span style="color: #6600EE; font-weight: bold">0.5</span><span style="color: #333333">*</span>random<span style="color: #333333">.</span>random()
        
        <span style="color: #008800; font-weight: bold">if</span> clss <span style="color: #333333">==</span> <span style="background-color: #fff0f0">&quot;squares&quot;</span>:
            path_data<span style="color: #333333">=</span> [
                (Path<span style="color: #333333">.</span>MOVETO, (basex, basey)), <span style="color: #888888">#move to base position of this image</span>
                (Path<span style="color: #333333">.</span>LINETO, (basex<span style="color: #333333">+</span>length, basey)), <span style="color: #888888">#draw line across to the right</span>
                (Path<span style="color: #333333">.</span>LINETO, (basex<span style="color: #333333">+</span>length, basey<span style="color: #333333">+</span>length )), <span style="color: #888888">#draw line up</span>
                (Path<span style="color: #333333">.</span>LINETO, (basex, basey<span style="color: #333333">+</span>length)), <span style="color: #888888">#draw line back across to the left</span>
                (Path<span style="color: #333333">.</span>LINETO, (basex, basey)), <span style="color: #888888">#draw line back down to base postiion</span>
            ]
        <span style="color: #008800; font-weight: bold">else</span>: <span style="color: #888888">#triangles</span>
            path_data<span style="color: #333333">=</span> [
                (Path<span style="color: #333333">.</span>MOVETO, (basex, basey)), <span style="color: #888888">#move to base position of this image</span>
                (Path<span style="color: #333333">.</span>LINETO, (basex<span style="color: #333333">+</span>length, basey)), <span style="color: #888888">#draw line across to the right</span>
                (Path<span style="color: #333333">.</span>LINETO, ((basex<span style="color: #333333">+</span>length<span style="color: #333333">/</span><span style="color: #6600EE; font-weight: bold">2.</span>), 
                    basey<span style="color: #333333">+</span>(math<span style="color: #333333">.</span>sqrt(<span style="color: #6600EE; font-weight: bold">3.</span>)<span style="color: #333333">*</span>length<span style="color: #333333">/</span><span style="color: #6600EE; font-weight: bold">2.</span>))), <span style="color: #888888">#draw line to top of equilateral triangle</span>
                (Path<span style="color: #333333">.</span>LINETO, (basex, basey)), <span style="color: #888888">#draw line back to base position            </span>
            ]

        <span style="color: #888888">#get the path data in the right format for plotting</span>
        codes, verts <span style="color: #333333">=</span> <span style="color: #007020">zip</span>(<span style="color: #333333">*</span>path_data)
        path <span style="color: #333333">=</span> mpath<span style="color: #333333">.</span>Path(verts, codes)

        <span style="color: #888888">#add shade the interior of the shape</span>
        patch <span style="color: #333333">=</span> mpatches<span style="color: #333333">.</span>PathPatch(path, facecolor<span style="color: #333333">=</span><span style="background-color: #fff0f0">&#39;gray&#39;</span>, alpha<span style="color: #333333">=</span><span style="color: #6600EE; font-weight: bold">0.5</span>)
        ax<span style="color: #333333">.</span>add_patch(patch)
        
        <span style="color: #888888">#set the scale of the overlall plot</span>
        plt<span style="color: #333333">.</span>xlim([<span style="color: #0000DD; font-weight: bold">0</span>,<span style="color: #0000DD; font-weight: bold">1</span>])
        plt<span style="color: #333333">.</span>ylim([<span style="color: #0000DD; font-weight: bold">0</span>,<span style="color: #0000DD; font-weight: bold">1</span>])

        <span style="color: #888888">#swith off plotting of the axis (only draw the shapes)</span>
        plt<span style="color: #333333">.</span>axis(<span style="background-color: #fff0f0">&#39;off&#39;</span>)

        <span style="color: #888888">#set the number of inches in each dimension to one</span>
        <span style="color: #888888"># - we will control the number of pixels in the next command</span>
        fig<span style="color: #333333">.</span>set_size_inches(<span style="color: #0000DD; font-weight: bold">1</span>, <span style="color: #0000DD; font-weight: bold">1</span>)

        <span style="color: #888888"># save the figure to file in te directory corresponding to its class</span>
        <span style="color: #888888"># the dpi=size (dots per inch) part sets the overall number of pixels to the</span>
        <span style="color: #888888"># desired value</span>
        fig<span style="color: #333333">.</span>savefig(<span style="background-color: #fff0f0">&#39;data/train/&#39;</span><span style="color: #333333">+</span>clss<span style="color: #333333">+</span><span style="background-color: #fff0f0">&#39;/data&#39;</span><span style="color: #333333">+</span><span style="color: #007020">str</span>(i)<span style="color: #333333">+</span><span style="background-color: #fff0f0">&#39;.png&#39;</span>,dpi<span style="color: #333333">=</span>size)   
        <span style="color: #888888"># close the figure</span>
        plt<span style="color: #333333">.</span>close(fig)    
</pre></div>

You now have a selection of 4000 squares and 4000 triangles in the <code>train/squares</code>  and <code>train/triangles</code> directories respectively:

![Virtual Shapes]({{ site.url }}/files/shapes.png)

Now, we will move a quarter of these to the <code>validate/squares</code>  and <code>validate/triangles</code> directories:

<pre>
mv data/train/squares/data3*  data/validate/squares/.
mv data/train/triangles/data3*  data/validate/triangles/.
</pre>


<h3>Converting to TensorFlow format</h3>

Change into the data directory:
<pre>
cd data
</pre>

Create a file called <code>mylabels.txt</code> and write to it the names of our classes:
<pre>
squares
triangles
</pre>

Now, to convert our images to TensorFlow [TFRecord format](https://www.tensorflow.org/versions/r0.12/api_docs/python/python_io.html#tfrecords-format-details), we are going to just use the [build_image_data.py](https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/build_image_data.py) script that is bundled with the Inception TensorFlow model. Get this by clinking on the above link, and then File->Save in your browser.

We can just use this a "black box" to convert our data (but we get some insight as to what it is doing later when we read the data within TensorFlow). Run the following command
<pre>
python build_image_data.py --train_directory=./train --output_directory=./  \
--validation_directory=./validate --labels_file=mylabels.txt   \
--train_shards=1 --validation_shards=1 --num_threads=1 
</pre>

We have told the script where to find the input files, and labels, and it will create a file containing all training images <code>train-00000-of-00001</code> and another containing all validation images <code>validation-00000-of-00001</code> in TensorFlow TFRecord format. We can now use these to train and validate our model.

Now change back up to the top-level directory:
<pre>
cd ..
</pre>

<h3>Training the model</h3>

In this section will see how to read in the previously generated TensorFlow TFRecord data files, and train the model. Please see the comments in each of the code snippets below. The full script can be downloaded [here]({{ site.url }}/files/shapesorter.py). 

First, we import the required modules and set some parameters:

<!-- HTML generated using hilite.me --><div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">tensorflow</span> <span style="color: #008800; font-weight: bold">as</span> <span style="color: #0e84b5; font-weight: bold">tf</span>
<span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">sys</span>
<span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">numpy</span>

<span style="color: #888888">#number of classes is 2 (squares and triangles)</span>
nClass<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>

<span style="color: #888888">#simple model (set to True) or convolutional neural network (set to False)</span>
simpleModel<span style="color: #333333">=</span><span style="color: #007020">True</span>

<span style="color: #888888">#dimensions of image (pixels)</span>
height<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">32</span>
width<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">32</span>
</pre></div>

Now, we can define a function which instructs TensorFlow how to read the data:

<!-- HTML generated using hilite.me --><div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #888888"># Function to tell TensorFlow how to read a single image from input file</span>
<span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">getImage</span>(filename):
    <span style="color: #888888"># convert filenames to a queue for an input pipeline.</span>
    filenameQ <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>train<span style="color: #333333">.</span>string_input_producer([filename],num_epochs<span style="color: #333333">=</span><span style="color: #007020">None</span>)
 
    <span style="color: #888888"># object to read records</span>
    recordReader <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>TFRecordReader()

    <span style="color: #888888"># read the full set of features for a single example </span>
    key, fullExample <span style="color: #333333">=</span> recordReader<span style="color: #333333">.</span>read(filenameQ)

    <span style="color: #888888"># parse the full example into its&#39; component features.</span>
    features <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>parse_single_example(
        fullExample,
        features<span style="color: #333333">=</span>{
            <span style="background-color: #fff0f0">&#39;image/height&#39;</span>: tf<span style="color: #333333">.</span>FixedLenFeature([], tf<span style="color: #333333">.</span>int64),
            <span style="background-color: #fff0f0">&#39;image/width&#39;</span>: tf<span style="color: #333333">.</span>FixedLenFeature([], tf<span style="color: #333333">.</span>int64),
            <span style="background-color: #fff0f0">&#39;image/colorspace&#39;</span>: tf<span style="color: #333333">.</span>FixedLenFeature([], dtype<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>string,default_value<span style="color: #333333">=</span><span style="background-color: #fff0f0">&#39;&#39;</span>),
            <span style="background-color: #fff0f0">&#39;image/channels&#39;</span>:  tf<span style="color: #333333">.</span>FixedLenFeature([], tf<span style="color: #333333">.</span>int64),            
            <span style="background-color: #fff0f0">&#39;image/class/label&#39;</span>: tf<span style="color: #333333">.</span>FixedLenFeature([],tf<span style="color: #333333">.</span>int64),
            <span style="background-color: #fff0f0">&#39;image/class/text&#39;</span>: tf<span style="color: #333333">.</span>FixedLenFeature([], dtype<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>string,default_value<span style="color: #333333">=</span><span style="background-color: #fff0f0">&#39;&#39;</span>),
            <span style="background-color: #fff0f0">&#39;image/format&#39;</span>: tf<span style="color: #333333">.</span>FixedLenFeature([], dtype<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>string,default_value<span style="color: #333333">=</span><span style="background-color: #fff0f0">&#39;&#39;</span>),
            <span style="background-color: #fff0f0">&#39;image/filename&#39;</span>: tf<span style="color: #333333">.</span>FixedLenFeature([], dtype<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>string,default_value<span style="color: #333333">=</span><span style="background-color: #fff0f0">&#39;&#39;</span>),
            <span style="background-color: #fff0f0">&#39;image/encoded&#39;</span>: tf<span style="color: #333333">.</span>FixedLenFeature([], dtype<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>string, default_value<span style="color: #333333">=</span><span style="background-color: #fff0f0">&#39;&#39;</span>)
        })


    <span style="color: #888888"># now we are going to manipulate the label and image features</span>

    label <span style="color: #333333">=</span> features[<span style="background-color: #fff0f0">&#39;image/class/label&#39;</span>]
    image_buffer <span style="color: #333333">=</span> features[<span style="background-color: #fff0f0">&#39;image/encoded&#39;</span>]

    <span style="color: #888888"># Decode the jpeg</span>
    <span style="color: #008800; font-weight: bold">with</span> tf<span style="color: #333333">.</span>name_scope(<span style="background-color: #fff0f0">&#39;decode_jpeg&#39;</span>,[image_buffer], <span style="color: #007020">None</span>):
        <span style="color: #888888"># decode</span>
        image <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>image<span style="color: #333333">.</span>decode_jpeg(image_buffer, channels<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">3</span>)
    
        <span style="color: #888888"># and convert to single precision data type</span>
        image <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>image<span style="color: #333333">.</span>convert_image_dtype(image, dtype<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>float32)


    <span style="color: #888888"># cast image into a single array, where each element corresponds to the greyscale</span>
    <span style="color: #888888"># value of a single pixel. </span>
    <span style="color: #888888"># the &quot;1-..&quot; part inverts the image, so that the background is black.</span>

    image<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>reshape(<span style="color: #0000DD; font-weight: bold">1</span><span style="color: #333333">-</span>tf<span style="color: #333333">.</span>image<span style="color: #333333">.</span>rgb_to_grayscale(image),[height<span style="color: #333333">*</span>width])

    <span style="color: #888888"># re-define label as a &quot;one-hot&quot; vector </span>
    <span style="color: #888888"># it will be [0,1] or [1,0] here. </span>
    <span style="color: #888888"># This approach can easily be extended to more classes.</span>
    label<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>pack(tf<span style="color: #333333">.</span>one_hot(label<span style="color: #333333">-</span><span style="color: #0000DD; font-weight: bold">1</span>, nClass))

    <span style="color: #008800; font-weight: bold">return</span> label, image
</pre></div>

Notice the structure of the full example in terms of its component features. If you were to look into the [build_image_data.py](https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/build_image_data.py) script that we used above to write the files, you would see that it organises the features precisely in this arrangement. You can see that we just extracted the image and label features, and performed some manipulation to get them into the right format for training.

We can then, using this function: 

<!-- HTML generated using hilite.me --><div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #888888"># associate the &quot;label&quot; and &quot;image&quot; objects with the corresponding features read from </span>
<span style="color: #888888"># a single example in the training data file</span>
label, image <span style="color: #333333">=</span> getImage(<span style="background-color: #fff0f0">&quot;data/train-00000-of-00001&quot;</span>)

<span style="color: #888888"># and similarly for the validation data</span>
vlabel, vimage <span style="color: #333333">=</span> getImage(<span style="background-color: #fff0f0">&quot;data/validation-00000-of-00001&quot;</span>)

<span style="color: #888888"># associate the &quot;label_batch&quot; and &quot;image_batch&quot; objects with a randomly selected batch---</span>
<span style="color: #888888"># of labels and images respectively</span>
imageBatch, labelBatch <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>train<span style="color: #333333">.</span>shuffle_batch(
    [image, label], batch_size<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">100</span>,
    capacity<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2000</span>,
    min_after_dequeue<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">1000</span>)

<span style="color: #888888"># and similarly for the validation data </span>
vimageBatch, vlabelBatch <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>train<span style="color: #333333">.</span>shuffle_batch(
    [vimage, vlabel], batch_size<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">100</span>,
    capacity<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2000</span>,
    min_after_dequeue<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">1000</span>)

<span style="color: #888888"># interactive session allows inteleaving of building and running steps</span>
sess <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>InteractiveSession()

<span style="color: #888888"># x is the input array, which will contain the data from an image </span>
<span style="color: #888888"># this creates a placeholder for x, to be populated later</span>
x <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>placeholder(tf<span style="color: #333333">.</span>float32, [<span style="color: #007020">None</span>, width<span style="color: #333333">*</span>height])
<span style="color: #888888"># similarly, we have a placeholder for true outputs (obtained from labels)</span>
y_ <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>placeholder(tf<span style="color: #333333">.</span>float32, [<span style="color: #007020">None</span>, nClass])
</pre></div>

The <code>tf.train.shuffle_batch</code> function is being used to get a randomly selected batch of 100 images from the data set. The other parameters in this function call can be adjusted for performance as described [here](https://www.tensorflow.org/versions/r0.11/api_docs/python/io_ops.html#shuffle_batch).

We are now ready to define the model. First, the simple model  (adapted from "MNIST For ML Beginners"):

<!-- HTML generated using hilite.me --><div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #008800; font-weight: bold">if</span> simpleModel:
  <span style="color: #888888"># run simple model y=Wx+b given in TensorFlow &quot;MNIST&quot; tutorial</span>

  <span style="color: #008800; font-weight: bold">print</span> <span style="background-color: #fff0f0">&quot;Running Simple Model y=Wx+b&quot;</span>

  <span style="color: #888888"># initialise weights and biases to zero</span>
  <span style="color: #888888"># W maps input to output so is of size: (number of pixels) * (Number of Classes)</span>
  W <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>Variable(tf<span style="color: #333333">.</span>zeros([width<span style="color: #333333">*</span>height, nClass]))
  <span style="color: #888888"># b is vector which has a size corresponding to number of classes</span>
  b <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>Variable(tf<span style="color: #333333">.</span>zeros([nClass]))

  <span style="color: #888888"># define output calc (for each class) y = softmax(Wx+b)</span>
  <span style="color: #888888"># softmax gives probability distribution across all classes</span>
  y <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>nn<span style="color: #333333">.</span>softmax(tf<span style="color: #333333">.</span>matmul(x, W) <span style="color: #333333">+</span> b)
</pre></div>

and also the convolutional neural network (adapted from "Deep MNIST for Experts") 

<!-- HTML generated using hilite.me --><div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #008800; font-weight: bold">else</span>:
  <span style="color: #888888"># run convolutional neural network model given in &quot;Expert MNIST&quot; TensorFlow tutorial</span>

  <span style="color: #888888"># functions to init small positive weights and biases</span>
  <span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">weight_variable</span>(shape):
    initial <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>truncated_normal(shape, stddev<span style="color: #333333">=</span><span style="color: #6600EE; font-weight: bold">0.1</span>)
    <span style="color: #008800; font-weight: bold">return</span> tf<span style="color: #333333">.</span>Variable(initial)

  <span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">bias_variable</span>(shape):
    initial <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>constant(<span style="color: #6600EE; font-weight: bold">0.1</span>, shape<span style="color: #333333">=</span>shape)
    <span style="color: #008800; font-weight: bold">return</span> tf<span style="color: #333333">.</span>Variable(initial)

  <span style="color: #888888"># set up &quot;vanilla&quot; versions of convolution and pooling</span>
  <span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">conv2d</span>(x, W):
    <span style="color: #008800; font-weight: bold">return</span> tf<span style="color: #333333">.</span>nn<span style="color: #333333">.</span>conv2d(x, W, strides<span style="color: #333333">=</span>[<span style="color: #0000DD; font-weight: bold">1</span>, <span style="color: #0000DD; font-weight: bold">1</span>, <span style="color: #0000DD; font-weight: bold">1</span>, <span style="color: #0000DD; font-weight: bold">1</span>], padding<span style="color: #333333">=</span><span style="background-color: #fff0f0">&#39;SAME&#39;</span>)

  <span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">max_pool_2x2</span>(x):
    <span style="color: #008800; font-weight: bold">return</span> tf<span style="color: #333333">.</span>nn<span style="color: #333333">.</span>max_pool(x, ksize<span style="color: #333333">=</span>[<span style="color: #0000DD; font-weight: bold">1</span>, <span style="color: #0000DD; font-weight: bold">2</span>, <span style="color: #0000DD; font-weight: bold">2</span>, <span style="color: #0000DD; font-weight: bold">1</span>],
                          strides<span style="color: #333333">=</span>[<span style="color: #0000DD; font-weight: bold">1</span>, <span style="color: #0000DD; font-weight: bold">2</span>, <span style="color: #0000DD; font-weight: bold">2</span>, <span style="color: #0000DD; font-weight: bold">1</span>], padding<span style="color: #333333">=</span><span style="background-color: #fff0f0">&#39;SAME&#39;</span>)

  <span style="color: #008800; font-weight: bold">print</span> <span style="background-color: #fff0f0">&quot;Running Convolutional Neural Network Model&quot;</span>
  nFeatures1<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">32</span>
  nFeatures2<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">64</span>
  nNeuronsfc<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">1024</span>

  <span style="color: #888888"># use functions to init weights and biases</span>
  <span style="color: #888888"># nFeatures1 features for each patch of size 5x5</span>
  <span style="color: #888888"># SAME weights used for all patches</span>
  <span style="color: #888888"># 1 input channel</span>
  W_conv1 <span style="color: #333333">=</span> weight_variable([<span style="color: #0000DD; font-weight: bold">5</span>, <span style="color: #0000DD; font-weight: bold">5</span>, <span style="color: #0000DD; font-weight: bold">1</span>, nFeatures1])
  b_conv1 <span style="color: #333333">=</span> bias_variable([nFeatures1])
  
  <span style="color: #888888"># reshape raw image data to 4D tensor. 2nd and 3rd indexes are W,H, fourth </span>
  <span style="color: #888888"># means 1 colour channel per pixel</span>
  <span style="color: #888888"># x_image = tf.reshape(x, [-1,28,28,1])</span>
  x_image <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>reshape(x, [<span style="color: #333333">-</span><span style="color: #0000DD; font-weight: bold">1</span>,width,height,<span style="color: #0000DD; font-weight: bold">1</span>])
  
  
  <span style="color: #888888"># hidden layer 1 </span>
  <span style="color: #888888"># pool(convolution(Wx)+b)</span>
  <span style="color: #888888"># pool reduces each dim by factor of 2.</span>
  h_conv1 <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>nn<span style="color: #333333">.</span>relu(conv2d(x_image, W_conv1) <span style="color: #333333">+</span> b_conv1)
  h_pool1 <span style="color: #333333">=</span> max_pool_2x2(h_conv1)
  
  <span style="color: #888888"># similarly for second layer, with nFeatures2 features per 5x5 patch</span>
  <span style="color: #888888"># input is nFeatures1 (number of features output from previous layer)</span>
  W_conv2 <span style="color: #333333">=</span> weight_variable([<span style="color: #0000DD; font-weight: bold">5</span>, <span style="color: #0000DD; font-weight: bold">5</span>, nFeatures1, nFeatures2])
  b_conv2 <span style="color: #333333">=</span> bias_variable([nFeatures2])
  

  h_conv2 <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>nn<span style="color: #333333">.</span>relu(conv2d(h_pool1, W_conv2) <span style="color: #333333">+</span> b_conv2)
  h_pool2 <span style="color: #333333">=</span> max_pool_2x2(h_conv2)
  
  
  <span style="color: #888888"># denseley connected layer. Similar to above, but operating</span>
  <span style="color: #888888"># on entire image (rather than patch) which has been reduced by a factor of 4 </span>
  <span style="color: #888888"># in each dimension</span>
  <span style="color: #888888"># so use large number of neurons </span>

  <span style="color: #888888"># check our dimensions are a multiple of 4</span>
  <span style="color: #008800; font-weight: bold">if</span> (width<span style="color: #333333">%</span><span style="color: #0000DD; font-weight: bold">4</span> <span style="color: #000000; font-weight: bold">or</span> height<span style="color: #333333">%</span><span style="color: #0000DD; font-weight: bold">4</span>):
    <span style="color: #008800; font-weight: bold">print</span> <span style="background-color: #fff0f0">&quot;Error: width and height must be a multiple of 4&quot;</span>
    sys<span style="color: #333333">.</span>exit(<span style="color: #0000DD; font-weight: bold">1</span>)
  
  W_fc1 <span style="color: #333333">=</span> weight_variable([(width<span style="color: #333333">/</span><span style="color: #0000DD; font-weight: bold">4</span>) <span style="color: #333333">*</span> (height<span style="color: #333333">/</span><span style="color: #0000DD; font-weight: bold">4</span>) <span style="color: #333333">*</span> nFeatures2, nNeuronsfc])
  b_fc1 <span style="color: #333333">=</span> bias_variable([nNeuronsfc])
  
  <span style="color: #888888"># flatten output from previous layer</span>
  h_pool2_flat <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>reshape(h_pool2, [<span style="color: #333333">-</span><span style="color: #0000DD; font-weight: bold">1</span>, (width<span style="color: #333333">/</span><span style="color: #0000DD; font-weight: bold">4</span>) <span style="color: #333333">*</span> (height<span style="color: #333333">/</span><span style="color: #0000DD; font-weight: bold">4</span>) <span style="color: #333333">*</span> nFeatures2])
  h_fc1 <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>nn<span style="color: #333333">.</span>relu(tf<span style="color: #333333">.</span>matmul(h_pool2_flat, W_fc1) <span style="color: #333333">+</span> b_fc1)
  
  <span style="color: #888888"># reduce overfitting by applying dropout</span>
  <span style="color: #888888"># each neuron is kept with probability keep_prob</span>
  keep_prob <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>placeholder(tf<span style="color: #333333">.</span>float32)
  h_fc1_drop <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>nn<span style="color: #333333">.</span>dropout(h_fc1, keep_prob)
  
  <span style="color: #888888"># create readout layer which outputs to nClass categories</span>
  W_fc2 <span style="color: #333333">=</span> weight_variable([nNeuronsfc, nClass])
  b_fc2 <span style="color: #333333">=</span> bias_variable([nClass])
  
  <span style="color: #888888"># define output calc (for each class) y = softmax(Wx+b)</span>
  <span style="color: #888888"># softmax gives probability distribution across all classes</span>
  <span style="color: #888888"># this is not run until later</span>
  y<span style="color: #333333">=</span>tf<span style="color: #333333">.</span>nn<span style="color: #333333">.</span>softmax(tf<span style="color: #333333">.</span>matmul(h_fc1_drop, W_fc2) <span style="color: #333333">+</span> b_fc2)
</pre></div>

Now, before we start training we need to define the error, train step, correct prediction and accuracy (common to both models):

<!-- HTML generated using hilite.me --><div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #888888"># measure of error of our model</span>
<span style="color: #888888"># this needs to be minimised by adjusting W and b</span>
cross_entropy <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>reduce_mean(<span style="color: #333333">-</span>tf<span style="color: #333333">.</span>reduce_sum(y_ <span style="color: #333333">*</span> tf<span style="color: #333333">.</span>log(y), reduction_indices<span style="color: #333333">=</span>[<span style="color: #0000DD; font-weight: bold">1</span>]))

<span style="color: #888888"># define training step which minimises cross entropy</span>
train_step <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>train<span style="color: #333333">.</span>AdamOptimizer(<span style="color: #6600EE; font-weight: bold">1e-4</span>)<span style="color: #333333">.</span>minimize(cross_entropy)

<span style="color: #888888"># argmax gives index of highest entry in vector (1st axis of 1D tensor)</span>
correct_prediction <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>equal(tf<span style="color: #333333">.</span>argmax(y,<span style="color: #0000DD; font-weight: bold">1</span>), tf<span style="color: #333333">.</span>argmax(y_,<span style="color: #0000DD; font-weight: bold">1</span>))

<span style="color: #888888"># get mean of all entries in correct prediction, the higher the better</span>
accuracy <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>reduce_mean(tf<span style="color: #333333">.</span>cast(correct_prediction, tf<span style="color: #333333">.</span>float32))
</pre></div>

And now we are ready to initialise and run the training:

<!-- HTML generated using hilite.me --><div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%"><span style="color: #888888"># initialize the variables</span>
sess<span style="color: #333333">.</span>run(tf<span style="color: #333333">.</span>initialize_all_variables())

<span style="color: #888888"># start the threads used for reading files</span>
coord <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>train<span style="color: #333333">.</span>Coordinator()
threads <span style="color: #333333">=</span> tf<span style="color: #333333">.</span>train<span style="color: #333333">.</span>start_queue_runners(sess<span style="color: #333333">=</span>sess,coord<span style="color: #333333">=</span>coord)

<span style="color: #888888"># start training</span>
nSteps<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">1000</span>
<span style="color: #008800; font-weight: bold">for</span> i <span style="color: #000000; font-weight: bold">in</span> <span style="color: #007020">range</span>(nSteps):

    batch_xs, batch_ys <span style="color: #333333">=</span> sess<span style="color: #333333">.</span>run([imageBatch, labelBatch])

    <span style="color: #888888"># run the training step with feed of images</span>
    <span style="color: #008800; font-weight: bold">if</span> simpleModel:
      train_step<span style="color: #333333">.</span>run(feed_dict<span style="color: #333333">=</span>{x: batch_xs, y_: batch_ys})
    <span style="color: #008800; font-weight: bold">else</span>:
      train_step<span style="color: #333333">.</span>run(feed_dict<span style="color: #333333">=</span>{x: batch_xs, y_: batch_ys, keep_prob: <span style="color: #6600EE; font-weight: bold">0.5</span>})


    <span style="color: #008800; font-weight: bold">if</span> (i<span style="color: #333333">+</span><span style="color: #0000DD; font-weight: bold">1</span>)<span style="color: #333333">%</span><span style="color: #0000DD; font-weight: bold">100</span> <span style="color: #333333">==</span> <span style="color: #0000DD; font-weight: bold">0</span>: <span style="color: #888888"># then perform validation </span>

      <span style="color: #888888"># get a validation batch</span>
      vbatch_xs, vbatch_ys <span style="color: #333333">=</span> sess<span style="color: #333333">.</span>run([vimageBatch, vlabelBatch])
      <span style="color: #008800; font-weight: bold">if</span> simpleModel:
        train_accuracy <span style="color: #333333">=</span> accuracy<span style="color: #333333">.</span>eval(feed_dict<span style="color: #333333">=</span>{
          x:vbatch_xs, y_: vbatch_ys})
      <span style="color: #008800; font-weight: bold">else</span>:
        train_accuracy <span style="color: #333333">=</span> accuracy<span style="color: #333333">.</span>eval(feed_dict<span style="color: #333333">=</span>{
          x:vbatch_xs, y_: vbatch_ys, keep_prob: <span style="color: #6600EE; font-weight: bold">1.0</span>})
      <span style="color: #008800; font-weight: bold">print</span>(<span style="background-color: #fff0f0">&quot;step </span><span style="background-color: #eeeeee">%d</span><span style="background-color: #fff0f0">, training accuracy </span><span style="background-color: #eeeeee">%g</span><span style="background-color: #fff0f0">&quot;</span><span style="color: #333333">%</span>(i<span style="color: #333333">+</span><span style="color: #0000DD; font-weight: bold">1</span>, train_accuracy))


<span style="color: #888888"># finalise </span>
coord<span style="color: #333333">.</span>request_stop()
coord<span style="color: #333333">.</span>join(threads)
</pre></div>

By running the full script with the simple model (from "MNIST For ML Beginners"), you will see that the training accuracy is around 60-70%. So the model better than useless (where a useless model would be equivalent to an uneducated guess, which would result in 50% accuracy), but still not very high. However, now change 
<pre>
simpleModel=True
</pre>
to
<pre>
simpleModel=False
</pre> 
to run the convolutional neural network (from "Deep MNIST for Experts") and run again: you will see the accuracy increase to between 95% and 100%.


<h2>Further Work</h2>

<h3> Image Resolution </h3>
Increase the resolution of the images you create to, say, 128x128 pixels, and train using these larger images (remembering to set the size properly at the top of the training script). You should see similar behaviour (but the training time will be longer).

<h3> Importance of Validation Set </h3>
 See what happens when you train using squares for <i>both</i> classes. As expected, the accuracy should be around 50% (i.e. the ability to predict is no better a uneducated guess since there is no conceptual difference between the classes). Now, temporarily replace the line
<pre>
  vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
</pre>
with
<pre>
  vbatch_xs, vbatch_ys = sess.run([imageBatch, labelBatch])
</pre>
to use the training images themselves for validation. You will see that this is a bad idea since the accuracy rises significantly above 50%, which is misleading. This demonstrating the importance of using a separate set of images for validation of the model.

<h3> Number of Classes </h3>
Add more classes. Work out how to draw different shapes using the matplotlib script, and adjust the training script to be able to train a network with more classes. 















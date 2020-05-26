# MultiGarmentNetwork
Repo for **"Multi-Garment Net: Learning to Dress 3D People from Images, ICCV'19"**

Link to paper: https://arxiv.org/abs/1908.06903

### Update 01 May 2020:
Added more garments here [Part-2](https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset_02.zip)

## Dress SMPL body model with our Digital Wardrobe

1. Download digital wardrobe: [Part-1](https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip) and [Part-2](https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset_02.zip).
This dataset contains scans, SMPL registration, texture_maps, segmentation_maps and multi-mesh registered garments.

2. visualize_scan.py: Load scan and visualize texture and segmentation.

3. visualize_garments.py: Visualize random garment and coresponding SMPL model.

4. dress_SMPL.py: Load random garment and dress desired SMPL body with it.


## Pre-requisites for running MGN
The code has been tested in python 2.7, Tensorflow 1.13

Download the neutral SMPL model from http://smplify.is.tue.mpg.de/ and place it in the `assets` folder.
```
cp <path_to_smplify>/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl assets/neutral_smpl.pkl
```

Download and install DIRT: https://github.com/pmh47/dirt.

Download and install mesh packages for visualization: https://github.com/MPI-IS/mesh

This repo contains code to run pretrained MGN model.
Download saved weights from : https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/weights.zip

## Data preparation

If you want to process your own data, some pre-processing steps are needed:

1. Crop your images to 720x720. In our testing setup we used roughly centerd subjects at a distance of around 2m from the camer.
2. Run semantic segmentation on images. We used [PGN semantic segmentation](https://github.com/Engineering-Course/CIHP_PGN) and manual correction. Segment garments, Pants (65, 0, 65), Short-Pants (0, 65, 65), Shirt (145, 65, 0), T-Shirt (145, 0, 65) and Coat (0, 145, 65).
3. Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) body_25 for 2D joints.

Semantic segmentation and OpenPose keypoints form the input to MGN. See `assets/test_data.pkl` folder for sample data.

## Texture

The following code may be used to stitch a texture for the reconstruction: https://github.com/thmoa/semantic_human_texture_stitching

Cite us:
```
@inproceedings{bhatnagar2019mgn,
    title = {Multi-Garment Net: Learning to Dress 3D People from Images},
    author = {Bhatnagar, Bharat Lal and Tiwari, Garvita and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
    month = {Oct},
    organization = {{IEEE}},
    year = {2019},
}
```

## License

Copyright (c) 2019 Bharat Lal Bhatnagar, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **Multi-Garment Net: Learning to Dress 3D People from Images** paper in documents and papers that report on research using this Software.


### Shoutouts

Chaitanya Patel: code for interpenetration removal, Thiemo Alldieck: code for texture/segmentation
stitching and Verica Lazova: code for data anonymization.

We thank Twindom and RenderPeople for providing data for the project. 

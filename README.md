# CarND-Traffic-Sign-Classifier
Building a neural netwrok to classify 43 different types of German traffic sign



Dataset

# Loading the dataset

First thing's first: download the [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).

`$ wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip`.

The zip file contains 3 pickle files with training, validation and test images, all resized to 32x32 pixels. The zip file also contains a CSV file (signnames.csv) with the first column containing the class ID (an integer spanning 0-42), and the second column containing a descriptive name of the sign. Here are the first 4 rows:

ClassId	SignName
0	Speed limit (20km/h)
1	Speed limit (30km/h)
2	Speed limit (50km/h)
3	Speed limit (60km/h)

# RCNN-SelectiveSearch

Selective Search Algorithm 
* This is a prerequisite for the Region based CNN network
* We first load the RGB image
* Get a segmentation map of the RGB image using a famous undirected graph based segmentation method.
* The segmentation map groups similar pixels together based on their intensity values
* On the segmentation map, find the bounding box coordinates by looping through each of the pixels in the image.
* Use the intensity value as a metric, to update the bounding box coordinates.
* Remove any bounding boxes which have overlapping left and right coordinates
* Once an initial set of regions have been obtained, we will now start merging regions based on the similarlites in texture, color, shape and size of the regions
* We first transform our RGB image to HSV space
* We extract the corresponding pixels of a region in the HSV image.
* For each region in the HSV space, we compute the histogram of intensity values for the 3 channels and stack them up.
* To find similarity between the color histograms of two regions, for each bin, we find the minimum. Once the min at all bins have been found, we sum them up to get a histogram similarity score. 
* For texture analysis, we can either compute gaussian derivatives in 8 different directions or can use Local Binary Patterns. We find the minimum across the two regions and sum the result.
* For size, we compute the number of pixels inside the region. To find similarlity we sum the sizes of the two regions, normalise it and subtract it from 1.
* For shape, we compute the union of the bounding box between two regions and subtract it from both sizes of both the regions. 
* All the similarity scores are combined to form one score
* For every pair of regions, we compute the similarity scores
* Based on similarity scores, we now have to merge the regions
* As a first step, we get the two regions with the highest similarity score and merge them
* The resulting bounding box will be the union of the two, the resulting size will be the sum of two, The resulting histogram will be the weighted sum of the histograms of each of the regions, where the weights are the sizes. Normalise the histograms
* Delete all the similarlity scores for the regions where any one of the two above regions existed
* For the new merged region and all the deleted regions, find the similartiy scores. 
* Repeat the process untill all regions have been merged

Raw Input Image
 <img src="https://github.com/soumilchugh/RCNN-SelectiveSearch/blob/master/dog.jpg" height="300" width="200">
 Region Proposals
  <img src="https://github.com/soumilchugh/RCNN-SelectiveSearch/blob/master/output.jpg" height="300" width="200">

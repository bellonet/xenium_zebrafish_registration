### Pipeline in `crop_per_fish_2d_slices.py`:  

Terms:  
tile - a 2d region with the 6 (or less) fish.  
bbox- of a 2d slice of fish.  

#### Step 1: Extract Tiles - Only uses the DAPI channel
1. Read the full-resolution DAPI image  
2. Downsample 8× for faster detection 
3. Threshold at intensity > 0 to create a binary mask
4. Connected-component labeling to find distinct tissue regions
5. Scale bounding box coordinates back to full resolution
6. Split any boxes that exceed MAX_HEIGHT/WIDTH px in height or width (into halves) - those are two connected tiles
7. Discard boxes where max intensity is low (empty/dark regions)
8. Crop each bounding box from the full-resolution DAPI image and save as individual tile TIFFs
9. Save tile metadata CSV (coordinates, intensities)

#### Step 2: Detect Fish Bounding Boxes 
Works on each tile from Step 1, at full resolution.  
1. For each tile, apply Otsu thresholding
2. Binary dilation (30 iterations) to connect nearby regions
3. Connected-component labeling (skimage) to find individual fish blobs
4. Filter out small area components
5. Merge small bounding boxes into nearby large ones; keep them as-is if no large neighbour
6. Save visualization PNGs showing detected boxes (those are not yet unique fish numbers)
7. Save bounding box CSV (tile-local coordinates)
   
#### Step 3: Crop & Reorder Tiles 
Tightens the tiles around actual content, then establishes a consistent spatial ordering.   
1. For each tile, compute the union bounding box of all detected fish boxes 
2. Crop the tile to this tighter region
3. Compute global coordinates (tile origin + local offset) for each fish bbox
4. Reorder tiles spatially: group tiles into rows by vertical overlap, then sort rows bottom-to-top, and within each row left-to-right
5. Renumber tiles sequentially (1, 2, 3, ...) in this new order
6. Save cropped tile TIFFs and updated CSVs

#### Step 4: Tag Fish by Position
Assigns each detected fish a consistent ID (1–6)   
1. Compute bbox centers
2. Decide if this tile likely has 1 row or 2 rows of fish (based on count and vertical spread)
3. Rotation search: try angles from -25° to +25° (step 5°) and score each:
- For 2-row layouts: k-means (k=2) on y-coordinates, score = within-cluster variance + separation penalty. - For 1-row layouts: score = variance of y-coordinates
Pick the angle that minimizes the score (i.e. rows are most horizontally aligned)
4. Rotate the tile image and all bbox coordinates by the best angle
5. Assign fish IDs (strict): - if a row has exactly 3 fish, label them left-to-right as columns 0/1/2; top row = IDs 0–2, bottom row = IDs 3–5
- Infer extra IDs (column overlap): for any untagged fish, check if its x-range overlaps with a tagged fish's column — if it matches exactly one column, assign accordingly
6. Save visualization PNGs (rotated tiles with labeled boxes and red X's on untagged ones)
7. Output two CSVs: tagged fish and untagged fish 

#### Step 5: Save Individual Fish Crops - Now uses all 4 channels 
For each channel:  
1. Load the full-resolution image from the corresponding OME-TIFF file
2. For each tagged fish: use the global bbox coordinates (from Step 4) + 10 px margin to crop
3. Save as individual_fish/ch{N}/{fish_id}/2d/{tile_idx}.tif
4. Free the full-res image before loading the next channel
Result: each fish (1–6) gets its own folder with one TIF per tile/z-plane, for each channel


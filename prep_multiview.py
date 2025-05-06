import pycolmap

database_path = "./colmap_tmp/database.db"
image_path = "./colmap_tmp/images"
output_path = "./colmap_tmp/sparse"

pycolmap.extract_features(database_path, image_path, sift_options={"max_image_size": 4096, "max_num_features": 16384, "estimate_affine_shape": 1, "domain_size_pooling": 1})
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_path, output_path)
maps[0].write(output_path)

pycolmap.undistort_images(mvs_path, output_path, image_path)
pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)


workdir=$1
# python scripts/downsample_point.py data/multipleview/$workdir/Boat_Pose_t50.ply ./data/multipleview/$workdir/points3D_multipleview.ply

python LLFF/imgs2poses.py data/multipleview/$workdir/

cp data/multipleview/$workdir/poses_bounds.npy data/multipleview/$workdir/poses_bounds_multipleview.npy


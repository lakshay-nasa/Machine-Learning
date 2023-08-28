def triangulate(point_2d_left, point_2d_right, baseline, focal_length):
    depth = (baseline * focal_length) / (point_2d_left[0] - point_2d_right[0])
    x = depth * (point_2d_left[0] - focal_length) / focal_length
    y = depth * (point_2d_left[1] - focal_length) / focal_length
    return x, y, depth

# Example 2D points in left and right images
point_2d_left = [100, 200]
point_2d_right = [120, 210]

# Example baseline (distance between cameras) and focal length
baseline = 10  # Arbitrary value for explanation
focal_length = 500  # Example focal length

# Calculate 3D point using triangulation
x, y, depth = triangulate(point_2d_left, point_2d_right, baseline, focal_length)

print("Calculated 3D Point (x, y, depth):", x, y, depth)




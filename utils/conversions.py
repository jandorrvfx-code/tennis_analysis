def convert_pixels_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    # cross multiplication
    meters = (pixel_distance * reference_height_in_meters) / reference_height_in_pixels
    return meters

def convert_meters_to_pixels(meters, reference_height_in_meters, reference_height_in_pixels):
    # cross multiplication
    pixels = (meters * reference_height_in_pixels) / reference_height_in_meters
    return pixels

from PIL import Image
import os


for image_num in range(1, 216):

    if not os.path.exists(f"data/images/{image_num}a.webp") or not os.path.exists(f"data/images/{image_num}b.webp"):
        continue

    # Paths to the input images
    image_a_path = f"data/images/{image_num}a.webp"
    image_b_path = f"data/images/{image_num}b.webp"
    output_path = f"data/images/{image_num}.png"

    # Open both images
    img_a = Image.open(image_a_path)
    img_b = Image.open(image_b_path)

    # Convert to RGB if necessary (handles RGBA, P, etc.)
    if img_a.mode != 'RGB':
        img_a = img_a.convert('RGB')
    if img_b.mode != 'RGB':
        img_b = img_b.convert('RGB')

    # Get dimensions
    width_a, height_a = img_a.size
    width_b, height_b = img_b.size

    # Calculate the dimensions of the combined image
    # Width should be the maximum of both widths to preserve original resolutions
    # Height should be the sum of both heights
    combined_width = max(width_a, width_b)
    combined_height = height_a + height_b

    # Create a new image with the combined dimensions
    # Use RGB mode for PNG output
    combined_img = Image.new('RGB', (combined_width, combined_height), color='white')

    # Paste 14a on top (at position 0, 0)
    combined_img.paste(img_a, (0, 0))

    # Paste 14b below 14a (at position 0, height_a)
    # Center horizontally if widths differ
    x_offset = (combined_width - width_b) // 2
    combined_img.paste(img_b, (x_offset, height_a))

    # Save as PNG
    combined_img.save(output_path, 'PNG')
    print(f"Combined image saved to {output_path}")
    print(f"Original resolutions preserved:")
    print(f"  {image_num}a.webp: {width_a}x{height_a}")
    print(f"  {image_num}b.webp: {width_b}x{height_b}")
    print(f"  Combined: {combined_width}x{combined_height}")

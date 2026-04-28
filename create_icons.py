"""
PWA Icon Generation Script
Generates simple seat icons as PNG files.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    """Generate PWA icon"""
    # Background color (green)
    img = Image.new('RGB', (size, size), color='#4CAF50')
    draw = ImageDraw.Draw(img)
    
    # Draw white circle (representing a seat)
    margin = size // 6
    draw.ellipse(
        [margin, margin, size-margin, size-margin],
        fill='white',
        outline='white'
    )
    
    # Add text
    try:
        # Calculate font size
        font_size = size // 3
        # Use system default font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = "S"
        
        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        position = ((size - text_width) // 2, (size - text_height) // 2 - font_size // 6)
        
        # Draw text
        draw.text(position, text, fill='#4CAF50', font=font)
    except Exception as e:
        print(f"Failed to add text: {e}")
    
    # Save
    img.save(filename)
    print(f"Icon created: {filename}")

# Create directory
os.makedirs('static/icons', exist_ok=True)

# Generate icons
create_icon(192, 'static/icons/icon-192.png')
create_icon(512, 'static/icons/icon-512.png')

print("\nPWA icons generated successfully!")
print("Location: static/icons/")
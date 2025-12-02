from PIL import Image, ImageDraw


logo1 = Image.open("streamlit_logo.jpeg")
logo2 = Image.open("python_logo.jpeg")
logo3 = Image.open("github.jpeg")

# Set specific width for each logo (adjust these values as needed)
logo1_width = 200   # Adjust this for first logo
logo2_width = 200   # Adjust this for second logo
logo3_width = 200   # Adjust this for third logo

# Calculate heights to maintain aspect ratio
logo1_height = 200
logo2_height = 200
logo3_height = 200

# Resize logos with specific widths
logo1 = logo1.resize((logo1_width, logo1_height))
logo2 = logo2.resize((logo2_width, logo2_height))
logo3 = logo3.resize((logo3_width, logo3_height))

# Use the tallest logo height for the canvas
max_height = max(logo1_height, logo2_height, logo3_height)

# Spacing between logos
spacing = 10

# Calculate total width
total_width = logo1_width + logo2_width + logo3_width + (2 * spacing)

# Create a new blank image with transparent background
combined = Image.new('RGBA', (total_width, max_height), (0, 0, 0, 0))

# Convert JPEGs to RGBA if needed
if logo1.mode != 'RGBA':
    logo1 = logo1.convert('RGBA')
if logo2.mode != 'RGBA':
    logo2 = logo2.convert('RGBA')
if logo3.mode != 'RGBA':
    logo3 = logo3.convert('RGBA')

# Paste logos vertically centered with spacing between them
x_position = 0

# Center logo1 vertically
y_offset1 = (max_height - logo1_height) // 2
combined.paste(logo1, (x_position, y_offset1), logo1)

x_position += logo1_width + spacing

# Center logo2 vertically
y_offset2 = (max_height - logo2_height) // 2
combined.paste(logo2, (x_position, y_offset2), logo2)

x_position += logo2_width + spacing

# Center logo3 vertically
y_offset3 = (max_height - logo3_height) // 2
combined.paste(logo3, (x_position, y_offset3), logo3)

# Save as PNG
combined.save("combined_logo.png")
print(f"Combined logo created!")
print(f"Logo 1: {logo1_width}px wide")
print(f"Logo 2: {logo2_width}px wide")
print(f"Logo 3: {logo3_width}px wide")
print(f"Spacing: {spacing}px")
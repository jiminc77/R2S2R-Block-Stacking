import cv2
import numpy as np
from cv2 import aruco
import os
import sys

# Constants
MM_TO_INCH = 0.0393701
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

def get_aruco_generator():
    """
    Returns the appropriate ArUco marker generation function 
    depending on the installed OpenCV version.
    """
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker
    elif hasattr(aruco, "drawMarker"):
        return aruco.drawMarker
    else:
        print("Error: Could not find 'generateImageMarker' or 'drawMarker' in cv2.aruco.")
        print(f"OpenCV Version: {cv2.__version__}")
        print("Available attributes:", dir(aruco))
        sys.exit(1)

def generate_aruco_markers_a4(
    marker_size_mm=40,
    border_size_mm=5,
    dpi=300,
    output_dir="aruco_markers"
):
    """
    Generate 16 ArUco markers (IDs 0-15) placed on an A4 sheet.
    
    Layout: 4x4 Grid
    Block 1: IDs 0-3
    Block 2: IDs 4-7
    Block 3: IDs 8-11
    Block 4: IDs 12-15
    """
    os.makedirs(output_dir, exist_ok=True)
    generator_func = get_aruco_generator()

    # Use 6x6 ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Calculate pixel sizes
    marker_size_px = int(marker_size_mm * MM_TO_INCH * dpi)
    border_size_px = int(border_size_mm * MM_TO_INCH * dpi)
    total_size_px = marker_size_px + (2 * border_size_px)
    
    # A4 size in pixels
    a4_width_px = int(A4_WIDTH_MM * MM_TO_INCH * dpi)
    a4_height_px = int(A4_HEIGHT_MM * MM_TO_INCH * dpi)
    
    # Create white A4 canvas
    a4_canvas = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255

    print(f"--- Configuration ---")
    print(f"Canvas: A4 ({a4_width_px}x{a4_height_px} px)")
    print(f"Marker Size: {marker_size_mm}mm ({marker_size_px} px)")
    print(f"Border Size: {border_size_mm}mm ({border_size_px} px)")
    print(f"Block Size: {marker_size_mm + 2*border_size_mm}mm ({total_size_px} px)")

    # Layout: 4x4 Grid centered on page
    cols, rows = 4, 4
    content_width_px = total_size_px * cols
    content_height_px = total_size_px * rows
    
    start_x = (a4_width_px - content_width_px) // 2
    start_y = (a4_height_px - content_height_px) // 2

    print(f"Layout: {cols}x{rows} Grid")
    print(f"Margins: X={start_x}px, Y={start_y}px")

    unique_id_counter = 0
    
    for block_id in range(4): # 4 Blocks
        for face_id in range(4): # 4 Faces per block
            
            # Grid position
            col = unique_id_counter % cols
            row = unique_id_counter // cols
            
            x = start_x + (col * total_size_px)
            y = start_y + (row * total_size_px)

            # Generate raw marker
            marker_img = generator_func(
                aruco_dict,
                unique_id_counter, 
                marker_size_px
            )

            # Add white border
            marker_with_border = cv2.copyMakeBorder(
                marker_img,
                border_size_px, border_size_px, border_size_px, border_size_px,
                cv2.BORDER_CONSTANT,
                value=255
            )
            
            # Add a thin rectangular outline for cutting
            final_marker = marker_with_border.copy()
            cv2.rectangle(
                final_marker, 
                (0, 0), 
                (final_marker.shape[1]-1, final_marker.shape[0]-1), 
                (0, 0, 0), 
                1
            )
            
            h, w = final_marker.shape
            
            # Place on canvas
            a4_canvas[y:y+h, x:x+w] = final_marker
            
            # Add label: B{block}(ID{id})
            text = f"B{block_id+1}(ID{unique_id_counter})"
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            text_x = x + (w - text_size[0]) // 2
            text_y = y + border_size_px - 10 
            
            cv2.putText(
                a4_canvas, 
                text, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                150, 
                thickness
            )

            unique_id_counter += 1

    # Save PNG
    png_path = f"{output_dir}/aruco_markers_A4.png"
    cv2.imwrite(png_path, a4_canvas)
    print(f"Saved PNG: {png_path}")

    # Save PDF
    try:
        import img2pdf
        pdf_path = f"{output_dir}/aruco_markers_A4.pdf"
        # Correctly specify layout function for fixed DPI
        layout_function = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(png_path, layout_fun=layout_function))
        print(f"Saved PDF: {pdf_path}")
    except ImportError:
        print("Warning: 'img2pdf' not installed. PDF generation skipped.")

    print(f"\n--- Instructions ---")
    print(f"1. Print '{png_path}' or PDF on A4 paper.")
    print(f"2. Set printer scaling to '100%' or 'Actual Size'.")
    print(f"3. Cut along the black outlines (50x50mm).")

if __name__ == "__main__":
    generate_aruco_markers_a4()
import fitz  # PyMuPDF
import os


def pdf_to_images(pdf_path, output_folder):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    num_pages = pdf_document.page_count

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each page
    for page_num in range(num_pages):
        # Get the page
        page = pdf_document.load_page(page_num)

        # Render the page to an image
        pix = page.get_pixmap()

        # Save the image with numbering based on the slide number (page number + 1)
        image_path = os.path.join(output_folder, f"slide_{page_num + 1}.png")
        pix.save(image_path)

    print(f"Converted {num_pages} pages to images in '{output_folder}'")


# Example usage
pdf_path = '/Users/twang/Downloads/[055256] CÁCH TIẾP CẬN HIỆN ĐẠI TRONG XỬ LÝ NGÔN NGỮ TỰ NHIÊN/Slides/NLP-DL-Lecture3.pdf'  # Đường dẫn tới file PDF
output_folder = '/Users/twang/PycharmProjects/HCMUT-TIMETABLE/demo_image_lect3'  # Thư mục để lưu hình ảnh
pdf_to_images(pdf_path, output_folder)

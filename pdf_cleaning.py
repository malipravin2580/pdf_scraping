import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from openai import OpenAI
import argparse
from pathlib import Path
import re
from imagekitio import ImageKit
import time
import uuid
from datetime import datetime
import tenacity
import yaml
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions



class PDFDataCleaner:
    def __init__(self):
        # Set OpenAI API key
        self.api_key = "OpenAI API Key"
        self.client = OpenAI(api_key=self.api_key)
        
        # Set Tesseract path for Linux
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        # ImageKit configuration
        self.imagekit = ImageKit(
            public_key='public_RNJO26njKaTKXFMr8NnAtXEM1A4=',
            private_key='private_lWEu8p5x8jrZlZMEpXaVgjV9FHw=',
            url_endpoint='https://ik.imagekit.io/ucmwbtbls'
        )

    def create_output_directories(self, base_output_dir, pdf_name):
        """Create output directories within a folder named after the PDF."""
        pdf_output_dir = os.path.join(base_output_dir, pdf_name)
        markdown_dir = os.path.join(pdf_output_dir, "markdown")
        images_dir = os.path.join(pdf_output_dir, "images")
        debug_dir = os.path.join(pdf_output_dir, "debug")
        os.makedirs(markdown_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        return markdown_dir, images_dir, debug_dir, pdf_output_dir

    def extract_images_from_pdf(self, pdf_path: str, images_dir: str, page_num: int, chapter_name: str) -> list:
        """Extract images from PDF page and save them."""
        try:
            pdf_document = fitz.open(pdf_path)
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            saved_images = []

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    if not base_image:
                        print(f"Warning: Could not extract image {img_index + 1} from page {page_num + 1}")
                        continue
                        
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Validate image data
                    if not image_bytes or len(image_bytes) < 100:  # Basic size check
                        print(f"Warning: Invalid image data for image {img_index + 1} on page {page_num + 1}")
                        continue
                        
                    unique_id = uuid.uuid4().hex[:6]
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"{unique_id}_{chapter_name}_page_{page_num + 1}_image_{img_index + 1}_{current_time}.{image_ext}"
                    image_path = os.path.join(images_dir, image_filename)

                    # Save image with error handling
                    try:
                        with open(image_path, "wb") as image_file:
                            image_file.write(image_bytes)
                            
                        # Verify the image was saved correctly
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                            saved_images.append({
                                'path': image_path,
                                'filename': image_filename,
                                'page': page_num + 1,
                                'index': img_index + 1
                            })
                            print(f"Successfully saved image: {image_filename}")
                        else:
                            print(f"Warning: Failed to save image {image_filename}")
                    except Exception as e:
                        print(f"Error saving image {image_filename}: {str(e)}")
                        
                except Exception as e:
                    print(f"Error processing image {img_index + 1} on page {page_num + 1}: {str(e)}")
                    continue

            pdf_document.close()
            return saved_images
        except Exception as e:
            print(f"Error extracting images: {str(e)}")
            return []

    def extract_text_with_ocr(self, pdf_path: str) -> list:
        """Extract text from the PDF page by page using OCR."""
        try:
            pdf_document = fitz.open(pdf_path)
            pages_text = []
            print(f"Total pages in PDF: {pdf_document.page_count}")

            for page_number in range(pdf_document.page_count):
                print(f"\nProcessing page {page_number + 1}...")
                page = pdf_document[page_number]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(
                    img, 
                    lang='eng',
                    config='--psm 3 --oem 3'
                )

                if text.strip():
                    pages_text.append({
                        'page_num': str(page_number + 1),
                        'text': text.strip()
                    })
                    print(f"Successfully extracted text from page {page_number + 1} (length: {len(text.strip())} characters)")
                else:
                    print(f"Warning: No text extracted from page {page_number + 1}")

            pdf_document.close()
            return pages_text
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return []

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: print(f"Retrying OpenAI API call (attempt {retry_state.attempt_number})...")
    )
    def clean_data_with_openai(self, raw_text: str, debug_dir: str, chapter_name: str) -> str:
        """Use OpenAI to clean and format the extracted text with timeout and retry."""
        try:
            # Save raw text for debugging
            safe_chapter_name = re.sub(r'[^\w\s-]', '', chapter_name).replace(' ', '_')
            debug_path = os.path.join(debug_dir, f"raw_{safe_chapter_name}.txt")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            print(f"Saved raw text to: {debug_path}")

            print(f"Sending text to OpenAI (length: {len(raw_text)} characters)")
            if not raw_text.strip():
                print("Warning: Empty text being sent to OpenAI")
                return ""
            
            # Split text into smaller chunks if too large (e.g., > 5000 characters)
            max_chunk_size = 5000
            if len(raw_text) > max_chunk_size:
                print(f"Text too large, splitting into chunks of {max_chunk_size} characters")
                text_chunks = [raw_text[i:i+max_chunk_size] for i in range(0, len(raw_text), max_chunk_size)]
            else:
                text_chunks = [raw_text]

            cleaned_chunks = []
            for i, chunk in enumerate(text_chunks):
                print(f"Processing chunk {i+1}/{len(text_chunks)} (length: {len(chunk)} characters)")
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Please format this text into clean markdown, strictly following these requirements:

### **1. REMOVE ALL of the following (do not include these in the output):**
- **Repeated headers or footers** (if they appear on multiple pages)
- **Contact information** (phone numbers, emails, websites)
- **Institution names and addresses**
- **Page numbers or running titles**
- **Author names and job titles**
- **Any text that appears at the top or bottom of every page** (likely headers or footers)

### **2. FORMAT the remaining content as:**
- Section headings (`## Section Heading`, e.g., `## Step View Procedure` or `## Procedure`)
- Subsections (`### Subsection Heading`, e.g., `### Steps` or `### Connector Removal`)
- Use numbered lists for procedural steps
- Format tables in markdown with proper alignment (e.g., columns like `Part Number`, `Description`, `Quantity`)
- Ensure proper lists and paragraph spacing
- For steps, include sub-points with bullet points (e.g., `- Note: ...`) if present
- Do not include the chapter title (e.g., 'Chapter X — Main Title') in the content, as it will be added separately

### **3. DETECT AND REMOVE REPETITIVE TEXT (HEADERS & FOOTERS):**
- Identify any text that appears at the **start or end of multiple pages** and remove it.

Here's the text to format:

{chunk}

Remember: Do not include any contact information, footers, institutional details, or the chapter title in the output."""
                        }
                    ],
                    max_tokens=1500,
                    timeout=30
                )
                elapsed_time = time.time() - start_time
                cleaned_text = response.choices[0].message.content.strip()
                print(f"Received response for chunk {i+1} from OpenAI (length: {len(cleaned_text)} characters, time: {elapsed_time:.2f}s)")
                cleaned_chunks.append(cleaned_text)

            # Combine cleaned chunks
            cleaned_text = "\n\n".join(cleaned_chunks)
            return cleaned_text
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            raise

    def group_by_chapters(self, pages_text: list) -> dict:
        """Group pages by chapters based on chapter headings."""
        chapters = {}
        current_chapter = None
        chapter_pattern = re.compile(r'Chapter\s*(\d+)\s*[—-]\s*([^\n:.{]+)', re.IGNORECASE)
        skip_pages = ['Preface', 'Terms and Conditions', 'Disclaimer', 'Operator Safety', 'Service Information', 'Alert Symbols']

        for page in pages_text:
            text = page['text']
            page_num = page['page_num']

            # Skip pages with unwanted sections
            for skip_section in skip_pages:
                if skip_section.lower() in text.lower():
                    print(f"Skipping page {page_num} as it contains: {skip_section}")
                    break
            else:
                # Check for chapter heading
                chapter_match = chapter_pattern.search(text)
                if chapter_match:
                    chapter_num = chapter_match.group(1)
                    chapter_title = chapter_match.group(2).strip()
                    current_chapter = f"Chapter {chapter_num} — {chapter_title}"
                    if current_chapter not in chapters:
                        chapters[current_chapter] = {
                            'pages': [],
                            'text': '',
                            'page_nums': [],
                            'simple_title': chapter_title,
                            'chapter_num': chapter_num
                        }
                    print(f"Found chapter: {current_chapter} on page {page_num}")
                else:
                    print(f"No chapter heading found on page {page_num}")

                if current_chapter:
                    chapters[current_chapter]['pages'].append(page)
                    chapters[current_chapter]['text'] += text + "\n\n"
                    chapters[current_chapter]['page_nums'].append(page_num)
                else:
                    print(f"Page {page_num} not assigned to any chapter")

        # Fallback: If no chapters are found, create a default chapter
        if not chapters:
            print("Warning: No chapters identified. Creating default chapter.")
            default_chapter = "Default Chapter"
            chapters[default_chapter] = {
                'pages': pages_text,
                'text': "\n\n".join(page['text'] for page in pages_text),
                'page_nums': [page['page_num'] for page in pages_text],
                'simple_title': default_chapter,
                'chapter_num': '0'
            }

        return chapters

    def split_markdown_into_chunks(self, markdown_content: str, chunk_size: int = 300) -> list:
        """Split markdown content into chunks while preserving section structure."""
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        lines = markdown_content.split('\n')
        
        for line in lines:
            words_in_line = len(line.split())
            
            if line.startswith('#') or line.startswith('##') or line.startswith('###'):
                if current_chunk and current_word_count + words_in_line > chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                
                current_chunk.append(line)
                current_word_count += words_in_line
                continue
            
            if current_word_count + words_in_line > chunk_size and current_chunk:
                if not (line.startswith('|') or line.startswith('-') or line.startswith('*')):
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
            
            current_chunk.append(line)
            current_word_count += words_in_line
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        cleaned_chunks = []
        for chunk in chunks:
            lines = chunk.split('\n')
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            if lines:
                cleaned_chunks.append('\n'.join(lines))
        
        return cleaned_chunks

    def upload_image_to_imagekit(self, image_path, image_filename):
        """Upload image to ImageKit and return the URL."""
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                return None
             # Configure upload options

            options = UploadFileRequestOptions(

                folder="pdf_images",

                is_private_file=False,

                use_unique_file_name=True

            )
            print(f"📤 Uploading image {image_filename} to ImageKit folder 'pdf_images'...")
            with open(image_path, "rb") as f:
                upload = self.imagekit.upload_file(
                    file=f,
                    file_name=image_filename,
                    options=options
                )
            if upload and hasattr(upload, 'url') and upload.url:
                print(f"Successfully uploaded image to ImageKit: {upload.url}")
                return upload.url
            else:
                print(f"ImageKit upload failed: {upload}")
                # Return local path as fallback
                return os.path.abspath(image_path)
        except Exception as e:
            print(f"Error uploading to ImageKit: {str(e)}")
            # Return local path as fallback
            return os.path.abspath(image_path)

    def process_pdf(self, pdf_path: str, output_dir: str) -> None:
        """Process the PDF to extract and clean data by chapters, with images inline and metadata."""
        print(f"\nProcessing PDF: {pdf_path}")
        print(f"Using API key: {self.api_key[:7]}...")
        
        pdf_name = Path(pdf_path).stem
        markdown_dir, images_dir, debug_dir, pdf_output_dir = self.create_output_directories(output_dir, pdf_name)
        print(f"Created PDF output directory: {pdf_output_dir}")
        print(f"Created markdown directory: {markdown_dir}")
        print(f"Created images directory: {images_dir}")
        print(f"Created debug directory: {debug_dir}")
        
        print(f"PDF name: {pdf_name}")
        
        pages_text = self.extract_text_with_ocr(pdf_path)
        print(f"Extracted {len(pages_text)} pages of text")
        
        if not pages_text:
            print("Error: No text was extracted from the PDF")
            return
        
        chapters = self.group_by_chapters(pages_text)
        print(f"Identified {len(chapters)} chapters")
        
        for chapter_name, chapter_data in chapters.items():
            print(f"\nProcessing chapter: {chapter_name}")
            chapter_text = chapter_data['text']
            page_nums = chapter_data['page_nums']
            pages = chapter_data['pages']
            simple_title = chapter_data['simple_title']
            chapter_num = chapter_data.get('chapter_num', '0')  # Fallback to '0' if chapter_num is missing
            
            # Log chapter number for debugging
            print(f"Chapter number: {chapter_num}")
            
            # Extract images for each page and map them to page numbers
            page_images = {}
            all_image_urls = {}  # Maps filenames to ImageKit URLs or local paths
            for page_num in page_nums:
                try:
                    page_num_int = int(page_num)
                    safe_chapter_name = re.sub(r'[^\w\s-]', '', simple_title).replace(' ', '_')
                    images = self.extract_images_from_pdf(pdf_path, images_dir, page_num_int - 1, safe_chapter_name)
                    page_images[page_num] = images
                    for img in images:
                        imagekit_url = self.upload_image_to_imagekit(img['path'], img['filename'])
                        all_image_urls[img['filename']] = imagekit_url if imagekit_url else img['path']
                except ValueError:
                    print(f"Warning: Invalid page number {page_num} for image extraction")
                    continue
            
            # Generate markdown content without chapter title in the middle
            full_content = ""
            for page in pages:
                page_num = page['page_num']
                page_cleaned = self.clean_data_with_openai(page['text'], debug_dir, f"{simple_title}_page_{page_num}")
                
                if page_cleaned.strip():
                    # Check if the page contains a table
                    table_match = re.search(r'\|.*\|.*\|', page_cleaned)
                    if table_match and page_num in page_images and page_images[page_num]:
                        # Split table into rows
                        table_lines = [line for line in page_cleaned.split('\n') if line.startswith('|')]
                        non_table_content = [line for line in page_cleaned.split('\n') if not line.startswith('|')]
                        
                        # Distribute images across table rows
                        images = page_images[page_num]
                        num_rows = len([line for line in table_lines if not line.startswith('|-')]) - 1  # Exclude header
                        num_images = len(images)
                        image_per_row = num_images // num_rows if num_rows > 0 else 0
                        extra_images = num_images % num_rows if num_rows > 0 else num_images
                        
                        table_content = []
                        image_index = 0
                        row_count = 0
                        
                        for line in table_lines:
                            table_content.append(line)
                            if not line.startswith('|-') and line.strip() != table_lines[0].strip():  # Skip header and separators
                                images_for_row = image_per_row + (1 if row_count < extra_images else 0)
                                for _ in range(images_for_row):
                                    if image_index < num_images:
                                        img = images[image_index]
                                        image_url = all_image_urls.get(img['filename'], img['path'])
                                        table_content.append(f"![{img['filename']}]({image_url})")
                                        image_index += 1
                                row_count += 1
                        
                        # Combine non-table content and table with images
                        full_content += '\n'.join(non_table_content + table_content) + "\n\n"
                    else:
                        # Non-table page: Distribute images across steps
                        lines = page_cleaned.split('\n')
                        step_content = []
                        images = page_images.get(page_num, [])
                        num_images = len(images)
                        image_index = 0
                        step_pattern = re.compile(r'^\d+\.\s+.*$')  # Matches numbered steps (e.g., "1. Text")
                        subpoint_pattern = re.compile(r'^\s*[-*]\s+.*$')  # Matches sub-points (e.g., "- Note: ...")
                        steps_and_subpoints = []
                        current_step_lines = []

                        # Group lines by steps and their sub-points
                        for line in lines:
                            if step_pattern.match(line):
                                if current_step_lines:
                                    steps_and_subpoints.append(current_step_lines)
                                    current_step_lines = []
                            current_step_lines.append(line)
                        if current_step_lines:
                            steps_and_subpoints.append(current_step_lines)

                        # Distribute images across steps and sub-points
                        num_steps = len(steps_and_subpoints)
                        images_per_step = num_images // num_steps if num_steps > 0 else 0
                        extra_images = num_images % num_steps if num_steps > 0 else num_images

                        for step_idx, step_lines in enumerate(steps_and_subpoints):
                            step_content.extend(step_lines)
                            images_for_step = images_per_step + (1 if step_idx < extra_images else 0)
                            
                            # Check if the step has sub-points
                            subpoint_count = sum(1 for line in step_lines if subpoint_pattern.match(line))
                            if subpoint_count > 0 and images_for_step > 1:
                                # Distribute images across sub-points if multiple images
                                subpoint_lines = [line for line in step_lines if subpoint_pattern.match(line)]
                                images_per_subpoint = images_for_step // (subpoint_count + 1)  # Include main step
                                extra_subpoint_images = images_for_step % (subpoint_count + 1)
                                subpoint_idx = 0
                                main_step_assigned = False

                                for line in step_lines:
                                    step_content.append(line)
                                    if not main_step_assigned and not subpoint_pattern.match(line):
                                        # Assign images to main step
                                        images_for_main = images_per_subpoint + (1 if subpoint_idx < extra_subpoint_images else 0)
                                        for _ in range(images_for_main):
                                            if image_index < num_images:
                                                img = images[image_index]
                                                image_url = all_image_urls.get(img['filename'], img['path'])
                                                step_content.append(f"![{img['filename']}]({image_url})")
                                                image_index += 1
                                        main_step_assigned = True
                                        subpoint_idx += 1
                                    elif subpoint_pattern.match(line):
                                        # Assign images to sub-point
                                        images_for_subpoint = images_per_subpoint + (1 if subpoint_idx < extra_subpoint_images else 0)
                                        for _ in range(images_for_subpoint):
                                            if image_index < num_images:
                                                img = images[image_index]
                                                image_url = all_image_urls.get(img['filename'], img['path'])
                                                step_content.append(f"![{img['filename']}]({image_url})")
                                                image_index += 1
                                        subpoint_idx += 1
                            else:
                                # Assign all images to the step
                                for _ in range(images_for_step):
                                    if image_index < num_images:
                                        img = images[image_index]
                                        image_url = all_image_urls.get(img['filename'], img['path'])
                                        step_content.append(f"![{img['filename']}]({image_url})")
                                        image_index += 1

                        # Add any remaining images
                        while image_index < num_images:
                            img = images[image_index]
                            image_url = all_image_urls.get(img['filename'])
                            if image_url:
                                # Ensure the image URL is properly formatted
                                if image_url.startswith('https://'):
                                    step_content.append(f"![{img['filename']}]({image_url})")
                                else:
                                    # For local paths, use relative path
                                    rel_path = os.path.relpath(image_url, markdown_dir)
                                    step_content.append(f"![{img['filename']}]({rel_path})")
                            image_index += 1

                        full_content += '\n'.join(step_content) + "\n\n"
            
            # Split the full content into chunks if needed
            chunks = self.split_markdown_into_chunks(full_content)
            
            for chunk_num, chunk in enumerate(chunks, 1):
                safe_chapter_name = re.sub(r'[^\w\s-]', '', simple_title).replace(' ', '_')
                first_page = page_nums[0] if page_nums else "unknown"
                md_filename = f"chapter_{chapter_num}_{safe_chapter_name}_{first_page}_{chunk_num}.md"
                md_path = os.path.join(markdown_dir, md_filename)
                
                # Extract image URLs referenced in this chunk and ensure they're properly formatted
                image_urls_in_chunk = []
                image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
                for match in image_pattern.finditer(chunk):
                    image_filename = match.group(1)
                    image_url = all_image_urls.get(image_filename)
                    if image_url:
                        if image_url.startswith('https://'):
                            image_urls_in_chunk.append(image_url)
                        else:
                            # For local paths, use relative path
                            rel_path = os.path.relpath(image_url, markdown_dir)
                            image_urls_in_chunk.append(rel_path)
              
                metadata = {
                    'metadata': {
                        'file': pdf_name,
                        'chapter': chapter_name,  # Add chapter name to metadata
                        'path': os.path.abspath(pdf_path),  # Add PDF file path
                        'url': os.path.join(output_dir, pdf_name, 'markdown', md_filename),
                        'page': first_page if len(page_nums) == 1 else f"{page_nums[0]}-{page_nums[-1]}" if page_nums else "unknown",
                        'images': image_urls_in_chunk
                    }
                }
                # metadata = {
                #     'metadata': {
                #         'file': pdf_name,  # Use pdf_name
                #         'url': os.path.join(output_dir, pdf_name, 'markdown', md_filename),
                #         'page': first_page if len(page_nums) == 1 else f"{page_nums[0]}-{page_nums[-1]}" if page_nums else "unknown",
                #         'images': image_urls_in_chunk
                #     }
                # }
                
                # Combine content and metadata (metadata at the end)
                content_with_metadata = f"# {chapter_name}\n\n{chunk}\n\n---\n{yaml.dump(metadata, allow_unicode=True)}"
                
                print(f"Attempting to write to: {md_path}")
                try:
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(content_with_metadata)
                    print(f"Created markdown chunk file: {md_path}")
                except Exception as e:
                    print(f"Error writing to markdown file: {str(e)}")
            

        print(f"\nProcessing complete! Markdown files saved in: {markdown_dir}")
        print(f"Images saved in: {images_dir}")

def main():
    parser = argparse.ArgumentParser(description='Clean data extracted from a PDF file by chapters.')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output-dir', default='output', help='Output directory for markdown files')

    args = parser.parse_args()
    cleaner = PDFDataCleaner()
    cleaner.process_pdf(args.pdf_path, args.output_dir)

if __name__ == "__main__":
    main()
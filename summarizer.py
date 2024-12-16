import os
import csv
import json
import torch
import arxiv
import sqlite3
import time  # Add this import
import warnings
import argparse
import fitz  # PyMuPDF
from datetime import datetime
from threading import Thread
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from transformers import TextIteratorStreamer, MllamaForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

class Summary(BaseModel):
    title: str
    main_problem: str = Field(..., max_length=512)
    approach: str = Field(..., max_length=1024)
    findings: str = Field(..., max_length=1024)
    impact: str = Field(..., max_length=1024)
    limitations: str = Field(..., max_length=1024)
    innovations: str = Field(..., max_length=1024)
    key_figures: str = Field(..., max_length=1024)
    primary_results: List[str] = Field(..., max_items=5)

class Tags(BaseModel):
    tags: List[str] = Field(..., min_items=5, max_items=5)

class Summarizer:
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", db_path: str = 'paper_summaries.db'):
        self.model = MllamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.db_path = db_path
        self.create_tables()
        print(f"Model is running on: {self.model.device}")
        
    def create_tables(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create the summaries table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS summaries
                    (arxiv_id TEXT PRIMARY KEY,
                    title TEXT,
                    main_problem TEXT,
                    approach TEXT,
                    findings TEXT,
                    impact TEXT,
                    limitations TEXT,
                    innovations TEXT,
                    key_figures TEXT,
                    primary_results TEXT,
                    tags TEXT,
                    publication_date TEXT,
                    upvotes INTEGER)''')
        
        # Ensure the hf_email table has the necessary columns
        c.execute('''CREATE TABLE IF NOT EXISTS hf_email
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_date TEXT,
                    title TEXT,
                    upvotes INTEGER,
                    arxiv_id TEXT,
                    downloaded INTEGER,
                    summarized INTEGER)''')
        
        conn.commit()
        conn.close()

    def get_unsummarized_papers(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT arxiv_id, title, email_date, upvotes
            FROM hf_email
            WHERE downloaded = 1 AND summarized = 0
        """)
        papers = [{'arxiv_id': row[0], 'title': row[1], 'date': row[2], 'upvotes': row[3]} for row in c.fetchall()]
        conn.close()
        return papers

    def extract_pdf_content(self, file_path: str) -> Dict[str, Any]:
        text_content = []
        images = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num, page in enumerate(doc):
                try:
                    # Extract text with error handling
                    try:
                        text_content.append(page.get_text())
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num}: {str(e)}")
                        text_content.append("")
                    
                    # Extract images with error handling
                    try:
                        for img_index, img in enumerate(page.get_images(full=True)):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                
                                images.append({
                                    "page": page_num,
                                    "index": img_index,
                                    "width": base_image["width"],
                                    "height": base_image["height"],
                                    "color_space": base_image["colorspace"],
                                })
                            except Exception as e:
                                print(f"Warning: Could not process image {img_index} on page {page_num}: {str(e)}")
                    except Exception as e:
                        print(f"Warning: Could not extract images from page {page_num}: {str(e)}")
                    
                except Exception as e:
                    print(f"Warning: Error processing page {page_num}: {str(e)}")
                    continue
                    
            doc.close()
            
            # If we couldn't extract any content, raise an error
            if not text_content and not images:
                raise Exception("Could not extract any content from the PDF")
                
            return {
                "text": "\n\n".join(text_content),
                "images": images
            }
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
            # Return empty content rather than failing completely
            return {
                "text": "Error: Could not extract content from PDF",
                "images": []
            }

    def generate_text_stream(self, prompt: str, max_new_tokens: int = 4096) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        try:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.7,
                temperature=0.2,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generation_kwargs['eos_token_id'] = self.tokenizer.encode("</explanation>")[-1]

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            for new_text in streamer:
                generated_text += new_text

            thread.join()
            
            torch.cuda.empty_cache()
            
            return generated_text
        except Exception as e:
            print(f"\nAn error occurred during text generation: {str(e)}")
            return None
        finally:
            del inputs
            torch.cuda.empty_cache()

    def summarize_text(self, content: Dict[str, Any], title: str, chunk_size: int = 32768, max_chunks: int = 10) -> Dict[str, Any]:
        text = content["text"]
        images = content["images"]
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        total_chunks = min(len(tokens) // chunk_size + (1 if len(tokens) % chunk_size else 0), max_chunks)
        print(f"Total chunks for this paper: {total_chunks}")
        
        chunk_summaries = []
        for i in range(0, min(len(tokens), chunk_size * max_chunks), chunk_size):
            chunk_number = i // chunk_size + 1
            print(f"Processing chunk {chunk_number}/{total_chunks}")
            
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            relevant_images = [img for img in images if img['page'] * chunk_size <= i < (img['page'] + 1) * chunk_size]
            image_info = "\n".join([f"Figure on page {img['page']}: Image {img['index']} (Size: {img['width']}x{img['height']}, Color Space: {img['color_space']})" for img in relevant_images])
            
            prompt = f"""Summarize the following excerpt from a scientific paper, including any mentioned figures and quantitative results, for a technically-minded audience:

            Text:
            {chunk_text}

            Figures:
            {image_info}

            Provide a concise summary that captures the main points using appropriate technical language. Focus on:
            1. The core idea or problem being addressed
            2. The approach or method used (including relevant technical terms)
            3. The key findings or results, with specific numerical data and statistics when available
            4. The potential impact or significance of the work in the broader technical context
            5. Any relevant information about the figures mentioned, including any data or results they present

            Be sure to include specific numbers, percentages, or other quantitative data when present in the text or figures.
            Aim for a balance between technical accuracy and general accessibility. Use technical terms where appropriate, but briefly explain any highly specialized concepts.

            Concise technical summary:"""

            chunk_summary = self.generate_text_stream(prompt, max_new_tokens=400)
            if chunk_summary:
                chunk_summaries.append(chunk_summary)

        print(f"Processed {len(chunk_summaries)} chunks")
        
        final_summary_prompt = f"""
        Based on the following summaries of different parts of a scientific paper titled "{title}", provide a structured summary of the entire paper.

        This summary is for a technical audience who may not be experts in this specific field but have a strong technical background. Use appropriate technical language and terms, but avoid highly specialized jargon without explanation.

        Summaries:
        {" ".join(chunk_summaries)}

        Provide a single, structured summary in the following JSON format:
        {{
          "title": "{title}",
          "main_problem": "The main problem or question the paper addresses and its relevance",
          "approach": "The key approach or methodology used, including notable technical aspects",
          "findings": "The most significant findings or results, with specific numerical data and statistics. Include key quantitative results from tables or figures if available",
          "impact": "The potential impact or implications of this research in the broader technical context",
          "limitations": "Any notable limitations or areas for future work",
          "innovations": "Any innovative techniques or technologies introduced (if applicable, otherwise state 'None specified')",
          "key_figures": "Brief description of key figures and their significance, including any important quantitative data they present",
          "primary_results": [
            "A concise list of the primary quantitative results",
            "Including specific numbers, percentages, or other relevant statistics",
            "From the study (provide 3-5 results)"
          ]
        }}
        Ensure that you include specific numerical results, percentages, or other quantitative data in your summary, especially in the 'findings' and 'primary_results' sections.
        Provide only one JSON object in your response. Do not include any text outside the JSON object.
        """

        final_summary_text = self.generate_text_stream(final_summary_prompt, max_new_tokens=2048)
        
        extracted_json = self.extract_json(final_summary_text)
        if extracted_json:
            try:
                validated_summary = Summary(**extracted_json)
                return validated_summary.dict()
            except ValidationError as e:
                print(f"Validation error: {str(e)}. Using partial data.")
                return extracted_json
        else:
            print("No valid JSON found. Using raw text.")
            return {"raw_summary": final_summary_text, "title": title}

    def generate_tags(self, text: Dict[str, Any]) -> List[str]:
        tag_prompt = f"""
        Based on the following summary of a scientific paper, generate exactly 5 tags that best represent the key topics, methods, technologies, or fields of study discussed in the paper. These tags should help categorize the paper and make it easily searchable for a technical audience.

        Prefer specific technical terms over general concepts when appropriate.

        Summary:
        {json.dumps(text, indent=2)}

        Generate exactly 5 Technical Tags in the following JSON format:
        {{
          "tags": [
            "tag1",
            "tag2",
            "tag3",
            "tag4",
            "tag5"
          ]
        }}
        Provide only the JSON object in your response. Do not include any text outside the JSON object.
        """

        tags_text = self.generate_text_stream(tag_prompt, max_new_tokens=100)
        
        extracted_json = self.extract_json(tags_text)
        if extracted_json:
            try:
                validated_tags = Tags(**extracted_json)
                return validated_tags.tags
            except ValidationError as e:
                print(f"Validation error in tags: {str(e)}. Using partial data.")
                return extracted_json.get("tags", ["error", "parsing", "tags", "partial", "data"])
        else:
            print("No valid JSON found in tags. Using default tags.")
            return ["error", "parsing", "tags", "default", "fallback"]

    def extract_json(self, text: str) -> Dict[str, Any]:
        import re
        json_candidates = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text)
        
        for candidate in json_candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        
        return None

    def insert_summary(self, arxiv_id: str, summary: Dict[str, Any], tags: List[str], publication_date: str, upvotes: int):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            INSERT OR REPLACE INTO summaries 
            (arxiv_id, title, main_problem, approach, findings, impact, limitations, 
             innovations, key_figures, primary_results, tags, publication_date, upvotes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            arxiv_id,
            summary['title'],
            summary['main_problem'],
            summary['approach'],
            summary['findings'],
            summary['impact'],
            summary['limitations'],
            summary['innovations'],
            summary['key_figures'],
            json.dumps(summary['primary_results']),
            json.dumps(tags),
            publication_date,
            upvotes
        ))
        
        # Update the hf_email table to mark the paper as summarized
        c.execute("""
            UPDATE hf_email 
            SET summarized = 1
            WHERE arxiv_id = ?
        """, (arxiv_id,))
        
        conn.commit()
        conn.close()

    def process_paper(self, paper: Dict[str, Any], pdf_dir: str):
        try:
            print(f"Processing paper: {paper['title']} (ArXiv ID: {paper['arxiv_id']})")
            pdf_path = os.path.join(pdf_dir, f"{paper['arxiv_id']}.pdf")
            
            if not os.path.exists(pdf_path):
                print(f"PDF file not found for {paper['arxiv_id']}. Skipping.")
                return

            document_content = self.extract_pdf_content(pdf_path)
            summary_dict = self.summarize_text(document_content, paper['title'])
            tags = self.generate_tags(summary_dict)
            self.insert_summary(paper['arxiv_id'], summary_dict, tags, paper['date'], paper['upvotes'])
            print(f"Summary and tags for paper '{paper['title']}' have been saved to the database")
        except Exception as e:
            print(f"Error processing paper '{paper['title']}': {e}")

def main(pdf_dir: str):
    summarizer = Summarizer()
    
    unsummarized_papers = summarizer.get_unsummarized_papers()

    if not unsummarized_papers:
        print("No unsummarized papers found.")
        return

    print(f"\nProcessing {len(unsummarized_papers)} unsummarized papers:")
    for paper in unsummarized_papers:
        summarizer.process_paper(paper, pdf_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize downloaded arXiv papers that haven't been summarized yet.")
    parser.add_argument("pdf_dir", help="Directory containing downloaded PDF files")
    args = parser.parse_args()

    main(args.pdf_dir)
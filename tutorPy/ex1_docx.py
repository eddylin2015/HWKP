"python-docx" 
from docx import Document

# Create a new Word document
doc = Document()

# Add a paragraph to the document
doc.add_paragraph("Hello, this is a paragraph added using python-docx!")

# Save the document
doc.save("example.docx")

print("Document created and saved as 'example.docx'")
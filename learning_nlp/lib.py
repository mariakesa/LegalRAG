import pandas as pd
from IPython.display import display, HTML

def get_data():
 return pd.read_json('/home/maria/Documents/LegalRAG/CUADv1.json')

def display_doc(data, doc_id):
    document = data['data'][doc_id]

    # Initialize an empty string to hold the HTML content
    html_content = f"<h2>Document Title: {document.get('title', 'No Title')}</h2>"

    # Iterate over each paragraph and wrap it in <p> tags
    for paragraph in document['paragraphs']:
        context = paragraph.get('context', '')
        html_content += f"<p>{context}</p>"

    # Display the formatted HTML
    display(HTML(html_content))

def print_question(dat, doc_id, q_id):
    return dat['data'][doc_id]['paragraphs'][0]['qas'][q_id]


def display_dependencies(doc):
    """
    Displays the dependency relations of a parsed Stanza document in a scrollable HTML table.
    
    Parameters:
    - doc: Parsed Stanza Document
    """
    for sent in doc.sentences:
        # Collect data into a list of dictionaries
        data = []
        for word in sent.words:
            data.append({
                'ID': word.id,
                'Text': word.text,
                'POS': word.upos,
                'Head': word.head,
                'Deprel': word.deprel
            })
        
        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Define CSS for scrollable div
        styles = """
        <style>
            .scrollable-table {
                max-height: 400px;
                overflow-y: scroll;
                border: 1px solid #ddd;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
                position: sticky;
                top: 0;
                z-index: 1;
            }
        </style>
        """
        
        # Convert DataFrame to HTML table
        html_table = df.to_html(index=False)
        
        # Combine styles with the table wrapped in a div
        scrollable_html = f"""
        {styles}
        <div class="scrollable-table">
            {html_table}
        </div>
        """
        
        # Display the formatted HTML
        display(HTML(scrollable_html))

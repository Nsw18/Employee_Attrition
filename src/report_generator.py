import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(insights, ml_insights, image_paths, output_filename="Attrition_Analysis_Report.pdf"):
    """
    Generates a professional PDF report containing the text insights and generated charts.
    """
    doc = SimpleDocTemplate(output_filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=22,
        spaceAfter=20,
        alignment=1, # Center
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        name='HeadingStyle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.black
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.spaceAfter = 8

    bullet_style = ParagraphStyle(
        name='BulletStyle',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=20,
        spaceAfter=6,
        bulletIndent=10
    )

    Story = []
    
    # Title Page
    Story.append(Spacer(1, 2 * 72))
    Story.append(Paragraph("Employee Attrition Analysis and Insight Generation", title_style))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("Data Analytics Report", ParagraphStyle(name='SubTitle', parent=styles['Heading2'], alignment=1)))
    Story.append(Spacer(1, 3 * 72))
    Story.append(PageBreak())

    # Section: Executive Summary & Insights
    Story.append(Paragraph("Executive Summary & Business Insights", heading_style))
    Story.append(Paragraph("This report presents a comprehensive analysis of employee attrition based on the HR analytics dataset. Below are the key data-driven insights derived from statistical methods and exploratory data analysis:", normal_style))
    
    for ins in insights:
        Story.append(Paragraph(f"• {ins}", bullet_style))
        
    Story.append(Spacer(1, 20))
    
    # Section: ML Validation Insights
    Story.append(Paragraph("Machine Learning Validation (Logistic Regression)", heading_style))
    Story.append(Paragraph("A Logistic Regression model was trained to identify the most significant drivers of attrition while maintaining high interpretability.", normal_style))
    
    for ml_ins in ml_insights:
        if ml_ins.startswith("---"):
            continue
        elif ml_ins.startswith("Top Factors"):
            Story.append(Spacer(1, 10))
            Story.append(Paragraph(ml_ins, styles['Heading4']))
        else:
            Story.append(Paragraph(ml_ins, bullet_style))
            
    Story.append(PageBreak())

    # Section: Visualizations
    Story.append(Paragraph("Key Visualizations", heading_style))
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                # Add image to PDF
                img = RLImage(img_path, width=400, height=300)
                Story.append(img)
                Story.append(Spacer(1, 20))
            except Exception as e:
                Story.append(Paragraph(f"Error loading image {img_path}: {str(e)}", normal_style))
    
    # Section: Recommendations
    Story.append(PageBreak())
    Story.append(Paragraph("Strategic Recommendations", heading_style))
    recs = [
        "1. Compensation Review: Given that lower income strongly correlates with attrition, HR should benchmark salaries against industry standards, particularly for high-risk roles.",
        "2. Overtime Management: High overtime significantly increases the likelihood of leaving. Implement policies to reduce mandatory overtime or provide better compensation/time-off in lieu.",
        "3. Departmental Interventions: Departments with the highest attrition should undergo targeted culture and workload assessments.",
        "4. Career Progression: Younger employees have a higher attrition rate. Establish clearer career paths and mentorship programs to retain junior talent."
    ]
    
    for rec in recs:
        Story.append(Paragraph(rec, normal_style))

    # Build the PDF
    try:
        doc.build(Story)
        return True, f"Report generated successfully: {output_filename}"
    except Exception as e:
        return False, f"Failed to generate report: {str(e)}"

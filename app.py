# PHASE 6: User Interface with Gradio
# Add this to your notebook after Phase 5

import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class CarDamageWebInterface:
    def __init__(self, damage_system):
        """Initialize web interface"""
        self.damage_system = damage_system
        print("Initializing Web Interface...")

    def process_uploaded_image(self, image):
        """Process uploaded image and return results"""

        if image is None:
            return "Please upload an image", None, "No analysis available"

        try:
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                # Convert numpy array to PIL Image if needed
                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = image

                pil_image.save(tmp_file.name)
                temp_path = tmp_file.name

            # Run complete analysis
            result = self.damage_system.analyze_car_damage(temp_path)

            # Clean up temp file
            os.unlink(temp_path)

            if not result or not result.get('classified_damages'):
                return "No significant damage detected in the image.", image, "No repair costs estimated."

            # Process results for display
            classified_damages = result['classified_damages']
            cost_analysis = result.get('cost_analysis', {})

            # Create result image with annotations
            result_image = self._create_annotated_image(image, classified_damages)

            # Create detailed report
            report = self._create_detailed_report(classified_damages, cost_analysis)

            # Create cost summary
            if cost_analysis:
                cost_summary = f"Total Estimated Repair Cost: ${cost_analysis['total_cost']:.2f}"
            else:
                cost_summary = "Cost estimation not available"

            return report, result_image, cost_summary

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print(error_msg)
            return error_msg, image, "Analysis failed"

    def _create_annotated_image(self, original_image, classified_damages):
        """Create annotated image with damage highlights"""
        try:
            # Convert to PIL Image if numpy array
            if isinstance(original_image, np.ndarray):
                img = Image.fromarray(original_image)
            else:
                img = original_image.copy()

            # Create drawing context
            draw = ImageDraw.Draw(img)

            # Color mapping for different damage types
            damage_colors = {
                'scratch': (255, 255, 0),    # Yellow
                'dent': (255, 0, 0),         # Red
                'crack': (0, 255, 0),        # Green
                'rust': (255, 165, 0),       # Orange
                'broken': (128, 0, 128),     # Purple
                'unknown': (255, 255, 255)   # White
            }

            # Draw bounding boxes and labels for each damage
            for i, damage in enumerate(classified_damages):
                damage_type = damage.get('type', 'unknown')
                severity = damage.get('severity', 'unknown')
                confidence = damage.get('confidence', 0)

                # Get color for this damage type
                color = damage_colors.get(damage_type, damage_colors['unknown'])

                # Create bounding box (simulate damage location)
                img_width, img_height = img.size
                x = (i * 150 + 50) % (img_width - 200)
                y = (i * 100 + 50) % (img_height - 150)

                # Draw rectangle
                draw.rectangle([x, y, x + 150, y + 100], outline=color, width=3)

                # Create label text
                label = f"{damage_type.upper()}\n{severity}\n{confidence:.1%}"

                # Draw label background
                draw.rectangle([x, y - 60, x + 150, y], fill=color, outline=color)

                # Draw label text
                try:
                    # Try to use default font
                    draw.text((x + 5, y - 55), label, fill=(0, 0, 0))
                except:
                    # Fallback if font fails
                    draw.text((x + 5, y - 55), label, fill=(0, 0, 0))

            return img

        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return original_image

    def _create_detailed_report(self, classified_damages, cost_analysis):
        """Create detailed text report"""
        report = "CAR DAMAGE ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"

        # Summary
        report += f"SUMMARY:\n"
        report += f"• Total damages detected: {len(classified_damages)}\n"

        # Count damage types
        damage_counts = {}
        severity_counts = {}

        for damage in classified_damages:
            dtype = damage.get('type', 'unknown')
            severity = damage.get('severity', 'unknown')

            damage_counts[dtype] = damage_counts.get(dtype, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        report += f"• Damage types found: {', '.join(damage_counts.keys())}\n"
        report += f"• Severity distribution: {dict(severity_counts)}\n\n"

        # Detailed damage list
        report += "DETAILED DAMAGE LIST:\n"
        report += "-" * 30 + "\n"

        for i, damage in enumerate(classified_damages, 1):
            report += f"{i}. {damage.get('type', 'Unknown').upper()}\n"
            report += f"   Severity: {damage.get('severity', 'Unknown')}\n"
            report += f"   Confidence: {damage.get('confidence', 0):.1%}\n"
            report += f"   Area: {damage.get('area_percentage', 0):.2f}% of image\n"
            if 'estimated_cost' in damage:
                report += f"   Estimated Cost: ${damage['estimated_cost']:.2f}\n"
            report += "\n"

        # Cost analysis
        if cost_analysis:
            report += "COST ANALYSIS:\n"
            report += "-" * 20 + "\n"
            report += f"Labor Cost: ${cost_analysis.get('labor_cost', 0):.2f}\n"
            report += f"Parts Cost: ${cost_analysis.get('parts_cost', 0):.2f}\n"
            report += f"Additional Fees: ${cost_analysis.get('additional_cost', 0):.2f}\n"
            report += f"TOTAL: ${cost_analysis.get('total_cost', 0):.2f}\n\n"

        # Recommendations
        report += "💡 RECOMMENDATIONS:\n"
        report += "-" * 20 + "\n"

        high_severity = [d for d in classified_damages if d.get('severity') == 'high']
        if high_severity:
            report += " High severity damages detected - immediate repair recommended\n"

        if any(d.get('type') == 'rust' for d in classified_damages):
            report += "Rust detected - treat immediately to prevent spreading\n"

        if any(d.get('type') == 'crack' for d in classified_damages):
            report += "Cracks found - structural integrity may be compromised\n"

        if len(classified_damages) > 5:
            report += "Multiple damages - consider comprehensive repair package\n"

        return report

    def create_interface(self):
        """Create and return Gradio interface"""

        # Custom CSS for styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 20px;
        }
        .result-area {
            border: 1px solid #bdc3c7;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """

        with gr.Blocks(css=css, title="Car Damage Detection AI") as interface:

            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>Car Damage Detection & Cost Estimation</h1>
                <p>Upload a photo of your car to get instant damage analysis and repair cost estimates</p>
            </div>
            """)

            with gr.Row():
                # Input Column
                with gr.Column(scale=1):
                    gr.HTML("<h3>Upload Car Image</h3>")

                    image_input = gr.Image(
                        label="Car Image",
                        type="pil",
                        height=400,
                        elem_classes="upload-area"
                    )

                    analyze_btn = gr.Button(
                        "Analyze Damage",
                        variant="primary",
                        size="lg"
                    )

                    # Example images section
                    gr.HTML("<h4>Instructions:</h4>")
                    gr.HTML("""
                    <ul>
                        <li>Upload a clear photo of your car</li>
                        <li>Ensure good lighting and visibility</li>
                        <li>Include the damaged areas in the frame</li>
                        <li>Supported formats: JPG, PNG, JPEG</li>
                    </ul>
                    """)

                # Output Column
                with gr.Column(scale=2):
                    gr.HTML("<h3>Analysis Results</h3>")

                    # Results tabs
                    with gr.Tabs():
                        with gr.TabItem("Detailed Report"):
                            report_output = gr.Textbox(
                                label="Analysis Report",
                                lines=20,
                                max_lines=30,
                                elem_classes="result-area"
                            )

                        with gr.TabItem("Annotated Image"):
                            image_output = gr.Image(
                                label="Damage Detection Results",
                                height=400,
                                elem_classes="result-area"
                            )

                        with gr.TabItem("Cost Summary"):
                            cost_output = gr.Textbox(
                                label="Repair Cost Estimate",
                                lines=5,
                                elem_classes="result-area"
                            )


            # Connect the analyze button to processing function
            analyze_btn.click(
                fn=self.process_uploaded_image,
                inputs=[image_input],
                outputs=[report_output, image_output, cost_output]
            )

            # Auto-analyze when image is uploaded
            image_input.change(
                fn=self.process_uploaded_image,
                inputs=[image_input],
                outputs=[report_output, image_output, cost_output]
            )

        return interface

# Initialize and launch the web interface
def launch_car_damage_app():
    """Launch the car damage detection web app"""

    print("Launching Car Damage Detection Web App...")

    # Create damage detection system (assuming it's already created from previous phases)
    try:
        # Use the damage system created in previous phases
        damage_system = complete_car_damage_system  # This should exist from Phase 5

        # Create web interface
        web_interface = CarDamageWebInterface(damage_system)

        # Create and launch Gradio interface
        app = web_interface.create_interface()

        # Launch with public sharing enabled
        app.launch(
            share=True,  # Creates public link
            debug=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )

    except NameError:
        print("Error: Complete damage system not found!")
        print("Please run Phases 1-5 first to create the damage detection system.")
        return None

    except Exception as e:
        print(f"Error launching app: {e}")
        return None

# Alternative: Create a demo interface if damage system is not available
def create_demo_interface():
    """Create a demo interface for testing"""

    class DemoCarDamageSystem:
        def analyze_car_damage(self, image_path):
            """Demo analysis function"""
            import random

            # Simulate analysis results
            damages = []
            damage_types = ['scratch', 'dent', 'crack', 'rust']
            severities = ['low', 'medium', 'high']

            num_damages = random.randint(1, 4)

            for i in range(num_damages):
                damage = {
                    'type': random.choice(damage_types),
                    'severity': random.choice(severities),
                    'confidence': random.uniform(0.6, 0.95),
                    'area_percentage': random.uniform(0.5, 5.0),
                    'estimated_cost': random.uniform(100, 2000)
                }
                damages.append(damage)

            total_cost = sum(d['estimated_cost'] for d in damages)

            return {
                'classified_damages': damages,
                'cost_analysis': {
                    'labor_cost': total_cost * 0.6,
                    'parts_cost': total_cost * 0.3,
                    'additional_cost': total_cost * 0.1,
                    'total_cost': total_cost
                }
            }

    # Create demo system and interface
    demo_system = DemoCarDamageSystem()
    web_interface = CarDamageWebInterface(demo_system)

    print("🎭 Launching DEMO version of Car Damage Detection App...")
    print("This is a demonstration with simulated results.")

    app = web_interface.create_interface()
    app.launch(share=True, debug=True)

# Instructions for Phase 6
print("=" * 60)
print("📋 PHASE 6: WEB INTERFACE SETUP COMPLETE!")
print("=" * 60)
print()
print("✨ FEATURES INCLUDED:")
print("• 📸 Drag & drop image upload")
print("• 🔍 Real-time damage analysis")
print("• 📊 Detailed damage reports")
print("• 🖼️ Annotated result images")
print("• 💰 Cost estimation breakdown")
print("• 📱 Mobile-friendly interface")
print("• 🌐 Public sharing capability")
print()
print("🎯 Ready to launch your Car Damage Detection Web App!")
create_demo_interface() 

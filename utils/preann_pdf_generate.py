"""
PDF report generation for the Pre-Annotation Quality Assessment pipeline.

Extracts all PDF/montage generation logic from the master script to keep it focused
on the analysis pipeline. Functions here handle:
- Activity and blurry image montage creation
- Image compression for PDF
- Full PDF report generation with all sections (executive summary, quality dashboard,
  blurry samples, diversity analysis, activity examples, cluster quality, coverage analysis)
"""

from pathlib import Path
from PIL import Image as PILImage, UnidentifiedImageError, ImageDraw, ImageFont
import numpy as np
import gc
import random
import shutil
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


def _load_image_safe(path: Path):
    """Load image safely, return None if corrupted."""
    try:
        img = PILImage.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError):
        return None


def create_activity_montage(image_paths, activity_id, max_samples, grid_cols, target_height):
    """
    Create a montage of sample images for an activity cluster with filenames below each image.
    """
    # Sample up to max_samples images
    n_samples = min(len(image_paths), max_samples)
    sampled_paths = random.sample(image_paths, n_samples) if len(image_paths) > max_samples else image_paths

    # Load images
    images = []
    filenames = []
    for img_path in sampled_paths:
        img = _load_image_safe(img_path)
        if img is not None:
            images.append(img)
            filenames.append(img_path.name)  # img_path is already a Path object

    if len(images) == 0:
        # Return placeholder
        placeholder = PILImage.new('RGB', (800, 200), color='gray')
        return placeholder

    # Resize all images to same size (maintain aspect ratio)
    resized_images = []
    for img in images:
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        resized = img.resize((new_width, target_height), PILImage.Resampling.LANCZOS)
        resized_images.append(resized)

    # Determine grid dimensions
    n_images = len(resized_images)
    n_rows = (n_images + grid_cols - 1) // grid_cols

    # Calculate montage dimensions (add 30px extra height per row for filename text with background)
    max_img_width = max(img.width for img in resized_images)
    montage_width = max_img_width * grid_cols + 10 * (grid_cols + 1)  # 10px padding
    montage_height = (target_height + 30) * n_rows + 10 * (n_rows + 1)  # Extra 30px for filename

    # Create montage canvas
    montage = PILImage.new('RGB', (montage_width, montage_height), color='white')
    draw = ImageDraw.Draw(montage)

    # Load font for filenames (bigger and bolder)
    try:
        font = ImageFont.truetype("arialbd.ttf", 11)  # Arial Bold 11pt
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 11)  # Fallback to regular Arial
        except:
            font = ImageFont.load_default()  # Final fallback

    # Paste images with filenames
    for idx, (img, filename) in enumerate(zip(resized_images, filenames)):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (max_img_width + 10) + 10
        y = row * (target_height + 30 + 10) + 10

        # Paste image
        montage.paste(img, (x, y))

        # Add filename below image with white background (show full filename)
        filename_display = filename

        # Calculate text position and background box
        text_bbox = draw.textbbox((0, 0), filename_display, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x + (img.width - text_width) // 2
        text_y = y + target_height + 5

        # Draw white background rectangle with slight padding
        padding = 3
        bg_box = [text_x - padding, text_y - padding,
                  text_x + text_width + padding, text_y + text_height + padding]
        draw.rectangle(bg_box, fill='white', outline='lightgray', width=1)

        # Draw filename text in black
        draw.text((text_x, text_y), filename_display, fill='black', font=font)

    return montage


def create_blurry_images_montage(blurry_image_cache, grid_cols, target_height):
    """
    Create a montage of blurry image samples using pre-cached images.
    Blurry = high gradient anisotropy (directional blur: motion blur / camera shake). Defocus is not detected.
    """
    if len(blurry_image_cache) == 0:
        return None

    # Use cached images directly (no loading needed)
    images = [item['image'] for item in blurry_image_cache]
    anisotropy_vals = [item['anisotropy'] for item in blurry_image_cache]
    brightness_vals = [item['brightness'] for item in blurry_image_cache]

    if len(images) == 0:
        return None

    # Resize all images to same size (maintain aspect ratio)
    resized_images = []
    for img in images:
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        resized = img.resize((new_width, target_height), PILImage.Resampling.LANCZOS)
        resized_images.append(resized)

    # Determine grid dimensions
    n_images = len(resized_images)
    n_rows = (n_images + grid_cols - 1) // grid_cols

    # Calculate montage dimensions
    max_img_width = max(img.width for img in resized_images)
    montage_width = max_img_width * grid_cols + 10 * (grid_cols + 1)  # 10px padding
    montage_height = (target_height + 45) * n_rows + 10 * (n_rows + 1)  # Extra 45px for 2 lines of text

    # Create montage canvas
    montage = PILImage.new('RGB', (montage_width, montage_height), color='white')

    # Paste images with anisotropy and brightness values
    draw = ImageDraw.Draw(montage)

    try:
        # Try to use a better font
        font = ImageFont.truetype("arial.ttf", 10)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    for idx, (img, anisotropy, brightness) in enumerate(zip(resized_images, anisotropy_vals, brightness_vals)):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (max_img_width + 10) + 10
        y = row * (target_height + 45 + 10) + 10

        # Paste image
        montage.paste(img, (x, y))

        # Add anisotropy and brightness values below image (2 lines)
        text_line1 = f"Aniso: {anisotropy:.2f}"
        text_line2 = f"Bright: {brightness:.1f}"

        text_bbox = draw.textbbox((0, 0), text_line1, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (img.width - text_width) // 2
        text_y = y + target_height + 5

        draw.text((text_x, text_y), text_line1, fill='red', font=font)
        draw.text((text_x, text_y + 15), text_line2, fill='blue', font=font)

    return montage


def compress_image_for_pdf(img_path, quality):
    """Compress PNG to JPEG for smaller PDF size."""
    try:
        img = PILImage.open(img_path)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        jpeg_path = img_path.with_suffix('.jpg')
        img.save(jpeg_path, 'JPEG', quality=quality, optimize=True)
        return jpeg_path
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to compress image {img_path}: {e}")
        # Return original path if compression fails
        return img_path


def generate_pdf_report(analysis_data, output_path, temp_dir, image_quality):
    """
    Generate comprehensive PDF report with compressed images.

    Args:
        image_quality: JPEG quality (1-100). Lower = smaller file. Default 75 is good balance.
    """
    print("\nBuilding PDF document...")
    print(f"  Image compression quality: {image_quality}%")
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           leftMargin=0.5*inch, rightMargin=0.5*inch)

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )

    # Add pages (blurry samples on Page 3 if exists, montages LAST)
    _add_executive_summary(story, analysis_data, title_style, heading_style, normal_style)
    _add_quality_dashboard(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality)
    _add_blurry_samples(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality)
    _add_diversity_analysis(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality)
    _add_cluster_quality(story, analysis_data, heading_style, subheading_style, normal_style)
    _add_coverage_analysis(story, analysis_data, temp_dir, heading_style, normal_style, image_quality)
    _add_activity_examples(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality)

    # Build PDF
    doc.build(story)

    # Cleanup temp files
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(1)
                gc.collect()
            else:
                print(f"\n⚠️  Warning: Could not delete temp directory {temp_dir}")
                print("    You may manually delete it after the process completes.")

    print(f"\n✅ PDF report generated: {output_path}")


def _add_executive_summary(story, analysis_data, title_style, heading_style, normal_style):
    """Page 1: Executive Summary"""
    story.append(Paragraph("PRE-ANNOTATION FRAME QUALITY REPORT", title_style))
    story.append(Paragraph("Engineer Dataset Assessment", heading_style))
    story.append(Spacer(1, 0.2*inch))

    stats = analysis_data['summary_stats']

    cfg = analysis_data.get('config', {})
    n_noise = stats.get('n_noise', 0)

    summary_data = [
        ['Metric', 'Value', 'Status'],
        ['Embedding Model', cfg.get('model_name', 'N/A'), ''],
        ['Scene Weight / Pose Weight',
         f"{cfg.get('scene_weight', 'N/A')} / {cfg.get('pose_weight', 'N/A')}", ''],
        ['', '', ''],
        ['Total Frames Analyzed', str(stats['total_frames']), ''],
        ['Valid Frames (loaded successfully)', str(stats['valid_frames']), ''],
        ['Activities Detected', str(stats['n_activities']),
         '✅ Good' if stats['n_activities'] >= 3 else '⚠ Limited diversity'],
        ['Noise / Outlier Frames', str(n_noise),
         '⚠ High' if (stats['total_frames'] > 0 and n_noise / stats['total_frames'] > 0.30) else '✅ OK'],
        ['', '', ''],
        ['Frames with Persons Detected',
         f"{stats['frames_with_persons']} ({stats['frames_with_persons']/stats['total_frames']*100:.1f}%)" if stats['total_frames'] > 0 else 'N/A (no frames)',
         ''],
        ['Max Persons in Single Frame', str(stats['max_persons_in_frame']), ''],
        ['', '', ''],
        ['Dark Frames (<80 brightness)', f"{stats['dark_count']} ({stats['dark_pct']:.1f}%)",
         '⚠ High' if stats['dark_pct'] > 60 else '✅ OK'],
        ['Medium Frames (80-175)', f"{stats['medium_count']} ({stats['medium_pct']:.1f}%)", ''],
        ['Bright Frames (>175)', f"{stats['bright_count']} ({stats['bright_pct']:.1f}%)", ''],
        ['', '', ''],
        ['Blurry Frames (motion blur, anisotropy >' + str(analysis_data['anisotropy_threshold']) + ')',
         f"{stats['blurry_count']} ({stats['blurry_pct']:.1f}%)",
         '⚠ High' if stats['blurry_pct'] > 40 else '✅ OK'],
        ['Non-Blurry Frames', f"{stats['sharp_count']} ({stats['sharp_pct']:.1f}%)", ''],
    ]

    summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    # Overall Quality Score
    quality_score = stats['quality_score']
    score_color = colors.green if quality_score >= 7 else (colors.orange if quality_score >= 5 else colors.red)

    story.append(Paragraph(f"<b>Overall Quality Score:</b> <font color='{score_color.hexval()}'>{quality_score:.1f}/10</font>",
                          ParagraphStyle('Score', parent=normal_style, fontSize=13, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.2*inch))

    # Key Recommendations
    story.append(Paragraph("Key Recommendations", ParagraphStyle('RecHeading', parent=normal_style, fontSize=12, fontName='Helvetica-Bold')))

    for rec in analysis_data['recommendations']:
        if rec.startswith('⚠'):
            bullet_style = ParagraphStyle('Warning', parent=normal_style,
                                         leftIndent=20, bulletIndent=10,
                                         textColor=colors.HexColor('#d32f2f'))
        elif rec.startswith('ℹ'):
            bullet_style = ParagraphStyle('Info', parent=normal_style,
                                         leftIndent=20, bulletIndent=10,
                                         textColor=colors.HexColor('#1976d2'))
        else:
            bullet_style = ParagraphStyle('Success', parent=normal_style,
                                         leftIndent=20, bulletIndent=10,
                                         textColor=colors.HexColor('#388e3c'))

        story.append(Paragraph(f"• {rec}", bullet_style))
        story.append(Spacer(1, 0.08*inch))

    story.append(PageBreak())


def _add_quality_dashboard(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality):
    """Page 2: Quality Metrics Dashboard"""
    story.append(Paragraph("Quality Metrics Dashboard", heading_style))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "This section shows the distribution of brightness and directional (motion) blur across all frames. Blur is measured by gradient anisotropy, which flags motion blur / camera shake but not defocus. "
        "Use these metrics to identify frames that should be dropped before annotation.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    quality_chart_path = compress_image_for_pdf(temp_dir / "quality_metrics.png", quality=image_quality)
    img = RLImage(str(quality_chart_path), width=7*inch, height=5.8*inch)
    story.append(img)
    story.append(Spacer(1, 0.2*inch))

    # Frames to drop recommendations
    story.append(Paragraph("Recommended Frames to Drop", subheading_style))

    stats = analysis_data['summary_stats']
    blurry_count = stats['blurry_count']
    drop_dark = stats['dark_count'] if stats['dark_pct'] > 70 else 0

    if blurry_count > 0 or drop_dark > 0:
        drop_text = f"Based on quality thresholds, consider removing:<br/>"
        if blurry_count > 0:
            drop_text += f"• <b>{blurry_count} blurry frames</b> (motion blur with anisotropy >{analysis_data['anisotropy_threshold']})<br/>"
        if drop_dark > 0:
            drop_text += f"• <b>{drop_dark} extremely dark frames</b> (brightness < 80, representing {stats['dark_pct']:.1f}% of dataset)"

        story.append(Paragraph(drop_text, normal_style))
    else:
        story.append(Paragraph("✅ No frames need to be dropped based on quality metrics.", normal_style))

    # Add green message if no blurry samples
    if blurry_count == 0:
        no_blurry_style = ParagraphStyle('NoBlurry', parent=normal_style,
                                        fontSize=11, fontName='Helvetica-Bold',
                                        textColor=colors.HexColor('#388e3c'))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("✅ No motion-blurred frames detected (anisotropy below threshold for all frames).", no_blurry_style))

    story.append(PageBreak())


def _add_blurry_samples(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality):
    """Page 3: Blurry Frames Visual Examples (only if blurry frames exist)"""
    stats = analysis_data['summary_stats']
    blurry_count = stats['blurry_count']

    blurry_montage_path_png = temp_dir / "blurry_images_montage.png"
    # Double check: file exists AND we have blurry frames (defensive against file system race conditions)
    if blurry_montage_path_png.exists() and blurry_count > 0:
        story.append(Paragraph("Blurry Frames - Visual Examples", heading_style))
        story.append(Spacer(1, 0.1*inch))

        blurry_montage_path = compress_image_for_pdf(blurry_montage_path_png, quality=image_quality)
        story.append(Paragraph(
            f"Below are sample motion-blurred frames ({blurry_count} flagged in total) that should be reviewed for removal. "
            f"These frames show directional blur (motion blur or camera shake); out-of-focus frames are not detected by this metric. "
            f"Anisotropy (motion blur metric) and brightness values are shown below each image.",
            normal_style
        ))
        story.append(Spacer(1, 0.15*inch))

        try:
            blurry_img_obj = PILImage.open(blurry_montage_path)
            img_width, img_height = blurry_img_obj.size

            max_width = 7 * inch
            width_scale = max_width / img_width
            scaled_height = img_height * width_scale

            # Intelligent splitting: if montage too tall, split at row boundaries
            max_page_height = 9.5 * inch  # Safe max height per page

            if scaled_height <= max_page_height:
                # Fits on one page - compress and add directly
                img = RLImage(str(blurry_montage_path), width=max_width, height=scaled_height)
                story.append(img)
            else:
                # Split at row boundaries (use actual grid_cols and cache limit from config)
                grid_cols = analysis_data.get('grid_cols_blurry', 4)
                cache_blurry_num_samples = analysis_data.get('cache_blurry_num_samples', 24)
                num_images = min(blurry_count, cache_blurry_num_samples)  # Actual samples cached in montage
                num_rows = (num_images + grid_cols - 1) // grid_cols

                # Calculate height per row
                row_height_px = img_height / num_rows
                row_height_scaled = row_height_px * width_scale

                # Calculate how many rows fit per page
                rows_per_page = max(1, int(max_page_height / row_height_scaled))

                # Split into chunks by whole rows
                num_chunks = (num_rows + rows_per_page - 1) // rows_per_page

                for chunk_idx in range(num_chunks):
                    # Calculate which rows to include
                    start_row = chunk_idx * rows_per_page
                    end_row = min((chunk_idx + 1) * rows_per_page, num_rows)

                    # Calculate pixel coordinates (crop at row boundaries)
                    top = int(start_row * row_height_px)
                    bottom = int(end_row * row_height_px)

                    # Crop the chunk
                    chunk = blurry_img_obj.crop((0, top, img_width, bottom))

                    # Save chunk temporarily as compressed JPEG
                    chunk_path = temp_dir / f"blurry_chunk_{chunk_idx}.jpg"
                    chunk.convert('RGB').save(chunk_path, 'JPEG', quality=90, optimize=True)

                    # Calculate chunk height after scaling
                    chunk_height = (bottom - top) * width_scale

                    # Add chunk to PDF
                    if chunk_idx > 0:
                        story.append(Spacer(1, 0.05*inch))
                        story.append(Paragraph(f"<i>(Continuation - Rows {start_row+1} to {end_row})</i>",
                                             ParagraphStyle('Continuation', parent=normal_style,
                                                          fontSize=9, alignment=TA_CENTER,
                                                          textColor=colors.grey)))
                        story.append(Spacer(1, 0.05*inch))

                    img = RLImage(str(chunk_path), width=max_width, height=chunk_height)
                    story.append(img)

                    # Add page break between chunks (except last one)
                    if chunk_idx < num_chunks - 1:
                        story.append(PageBreak())

                    del chunk  # Free memory

            # Explicit cleanup of blurry montage image
            del blurry_img_obj
            gc.collect()

        except Exception as e:
            story.append(Paragraph(f"⚠ Error loading blurry images montage: {e}", normal_style))
            if 'blurry_img_obj' in locals():
                del blurry_img_obj
                gc.collect()

        story.append(PageBreak())


def _add_diversity_analysis(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality):
    """Page 4: Diversity Analysis (or Page 3 if no blurry samples)"""
    story.append(Paragraph("Frame Diversity Analysis", heading_style))
    story.append(Spacer(1, 0.1*inch))

    stats = analysis_data['summary_stats']
    story.append(Paragraph(
        f"Frames were automatically grouped into <b>{stats['n_activities']} activity clusters</b> "
        f"based on semantic similarity (objects, positions, scenarios). "
        f"Each cluster represents a distinct type of work activity or scenario captured in the frames.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    umap_chart_path = compress_image_for_pdf(temp_dir / "umap_diversity.png", quality=image_quality)
    img = RLImage(str(umap_chart_path), width=7*inch, height=5.6*inch)
    story.append(img)
    story.append(Spacer(1, 0.2*inch))

    # Activity breakdown
    story.append(Paragraph("Activity Cluster Breakdown", subheading_style))

    activity_data = [['Activity Cluster', 'Frame Count', 'Percentage']]
    unique_activities = np.unique(analysis_data['activity_labels'])

    # Show clusters first, then noise
    cluster_activities = [a for a in unique_activities if a != -1]
    for activity_id in cluster_activities:
        count = np.sum(analysis_data['activity_labels'] == activity_id)
        pct = count / len(analysis_data['activity_labels']) * 100
        activity_data.append([f'Activity {activity_id}', str(count), f'{pct:.1f}%'])

    # Add noise row if present
    if -1 in unique_activities:
        count = np.sum(analysis_data['activity_labels'] == -1)
        pct = count / len(analysis_data['activity_labels']) * 100
        activity_data.append(['Noise/Outliers', str(count), f'{pct:.1f}%'])

    activity_table = Table(activity_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    activity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))

    story.append(activity_table)
    story.append(PageBreak())


def _add_activity_examples(story, analysis_data, temp_dir, heading_style, subheading_style, normal_style, image_quality):
    """Page 4: Activity Visual Examples (starts on fresh page)"""
    story.append(PageBreak())
    story.append(Paragraph("Activity Visual Examples", heading_style))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "Below are sample frames from each activity cluster to help you visually validate the clustering quality.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    unique_activities = np.unique(analysis_data['activity_labels'])
    cluster_activities = [a for a in unique_activities if a != -1]

    # Show regular clusters first
    for activity_id in cluster_activities:
        activity_name = analysis_data.get('activity_names', {}).get(activity_id, f'Activity {activity_id}')
        count = np.sum(analysis_data['activity_labels'] == activity_id)

        story.append(Paragraph(
            f"<b>Activity {activity_id}:</b> {activity_name} ({count} frames)",
            subheading_style
        ))
        story.append(Spacer(1, 0.1*inch))

        montage_path_png = temp_dir / f"activity_{activity_id}_montage.png"
        if montage_path_png.exists():
            montage_path = compress_image_for_pdf(montage_path_png, quality=image_quality)
            try:
                montage_img = PILImage.open(montage_path)
                img_width, img_height = montage_img.size

                max_width = 7 * inch
                width_scale = max_width / img_width
                scaled_height = img_height * width_scale

                # Intelligent splitting: if montage too tall, split at row boundaries
                max_page_height = 9.5 * inch

                if scaled_height <= max_page_height:
                    # Fits on one page
                    img = RLImage(str(montage_path), width=max_width, height=scaled_height)
                    story.append(img)
                else:
                    # Split at row boundaries (use actual grid_cols and max_samples from config)
                    grid_cols = analysis_data.get('grid_cols_activities', 4)
                    activity_num_samples = analysis_data.get('activity_num_samples', 20)
                    num_images = min(count, activity_num_samples)  # Actual samples used in montage
                    num_rows = (num_images + grid_cols - 1) // grid_cols

                    row_height_px = img_height / num_rows
                    row_height_scaled = row_height_px * width_scale

                    rows_per_page = max(1, int(max_page_height / row_height_scaled))
                    num_chunks = (num_rows + rows_per_page - 1) // rows_per_page

                    for chunk_idx in range(num_chunks):
                        start_row = chunk_idx * rows_per_page
                        end_row = min((chunk_idx + 1) * rows_per_page, num_rows)

                        top = int(start_row * row_height_px)
                        bottom = int(end_row * row_height_px)

                        chunk = montage_img.crop((0, top, img_width, bottom))
                        chunk_path = temp_dir / f"activity_{activity_id}_chunk_{chunk_idx}.jpg"
                        chunk.convert('RGB').save(chunk_path, 'JPEG', quality=90, optimize=True)

                        chunk_height = (bottom - top) * width_scale

                        if chunk_idx > 0:
                            story.append(Spacer(1, 0.05*inch))
                            story.append(Paragraph(f"<i>(Continuation - Rows {start_row+1} to {end_row})</i>",
                                                 ParagraphStyle('Continuation', parent=normal_style,
                                                              fontSize=9, alignment=TA_CENTER,
                                                              textColor=colors.grey)))
                            story.append(Spacer(1, 0.05*inch))

                        img = RLImage(str(chunk_path), width=max_width, height=chunk_height)
                        story.append(img)

                        if chunk_idx < num_chunks - 1:
                            story.append(PageBreak())

                        del chunk

                # Explicit cleanup of montage image
                del montage_img
                gc.collect()

                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"⚠ Error loading montage: {e}", normal_style))
                story.append(Spacer(1, 0.1*inch))
                if 'montage_img' in locals():
                    del montage_img
                    gc.collect()
        else:
            story.append(Paragraph("⚠ Montage not generated", normal_style))
            story.append(Spacer(1, 0.1*inch))

    # Show noise/outliers if present
    if -1 in unique_activities:
        n_noise = np.sum(analysis_data['activity_labels'] == -1)
        story.append(Paragraph(
            f"<b>Noise/Outliers:</b> Rare or unique frames ({n_noise} frames)",
            subheading_style
        ))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(
            "These frames did not fit into any activity cluster. They may represent rare activities, "
            "transitions between activities, or frames with poor quality/visibility.",
            normal_style
        ))
        story.append(Spacer(1, 0.1*inch))

        montage_path_png = temp_dir / f"activity_-1_montage.png"
        if montage_path_png.exists():
            montage_path = compress_image_for_pdf(montage_path_png, quality=image_quality)
            try:
                montage_img = PILImage.open(montage_path)
                img_width, img_height = montage_img.size

                max_width = 7 * inch
                width_scale = max_width / img_width
                scaled_height = img_height * width_scale

                # Intelligent splitting: if montage too tall, split at row boundaries
                max_page_height = 9.5 * inch

                if scaled_height <= max_page_height:
                    # Fits on one page
                    img = RLImage(str(montage_path), width=max_width, height=scaled_height)
                    story.append(img)
                else:
                    # Split at row boundaries (use actual grid_cols and outliers_num_samples from config)
                    grid_cols = analysis_data.get('grid_cols_activities', 4)
                    outliers_num_samples = analysis_data.get('outliers_num_samples', 50)
                    num_images = min(n_noise, outliers_num_samples)  # Actual samples used in outlier montage
                    num_rows = (num_images + grid_cols - 1) // grid_cols

                    row_height_px = img_height / num_rows
                    row_height_scaled = row_height_px * width_scale

                    rows_per_page = max(1, int(max_page_height / row_height_scaled))
                    num_chunks = (num_rows + rows_per_page - 1) // rows_per_page

                    for chunk_idx in range(num_chunks):
                        start_row = chunk_idx * rows_per_page
                        end_row = min((chunk_idx + 1) * rows_per_page, num_rows)

                        top = int(start_row * row_height_px)
                        bottom = int(end_row * row_height_px)

                        chunk = montage_img.crop((0, top, img_width, bottom))
                        chunk_path = temp_dir / f"outliers_chunk_{chunk_idx}.jpg"
                        chunk.convert('RGB').save(chunk_path, 'JPEG', quality=90, optimize=True)

                        chunk_height = (bottom - top) * width_scale

                        if chunk_idx > 0:
                            story.append(Spacer(1, 0.05*inch))
                            story.append(Paragraph(f"<i>(Continuation - Rows {start_row+1} to {end_row})</i>",
                                                 ParagraphStyle('Continuation', parent=normal_style,
                                                              fontSize=9, alignment=TA_CENTER,
                                                              textColor=colors.grey)))
                            story.append(Spacer(1, 0.05*inch))

                        img = RLImage(str(chunk_path), width=max_width, height=chunk_height)
                        story.append(img)

                        if chunk_idx < num_chunks - 1:
                            story.append(PageBreak())

                        del chunk

                # Explicit cleanup of montage image
                del montage_img
                gc.collect()

                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"⚠ Error loading noise montage: {e}", normal_style))
                story.append(Spacer(1, 0.1*inch))
                if 'montage_img' in locals():
                    del montage_img
                    gc.collect()

        story.append(Paragraph(
            f"<b>Total outlier frames:</b> {n_noise} — full list not shown in PDF to avoid overflow. "
            f"Re-run the script and choose to save outliers to a folder when prompted at the end.",
            ParagraphStyle('NoiseCount', parent=normal_style, fontSize=10,
                           textColor=colors.HexColor('#555555'))
        ))
        story.append(Spacer(1, 0.1*inch))

    story.append(PageBreak())


def _add_cluster_quality(story, analysis_data, heading_style, subheading_style, normal_style):
    """Page 5: Cluster Quality Metrics"""
    story.append(PageBreak())
    story.append(Paragraph("Cluster Quality Metrics", heading_style))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "These metrics assess the quality of the activity clustering. Higher Silhouette and Calinski-Harabasz "
        "scores indicate better-defined clusters, while lower Davies-Bouldin scores are better.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    cluster_quality = analysis_data['cluster_quality']

    # Metrics explanation table
    silhouette_val = f"{cluster_quality['silhouette']:.3f}" if cluster_quality['silhouette'] is not None else "N/A"
    calinski_val = f"{cluster_quality['calinski_harabasz']:.1f}" if cluster_quality['calinski_harabasz'] is not None else "N/A"
    davies_val = f"{cluster_quality['davies_bouldin']:.3f}" if cluster_quality['davies_bouldin'] is not None else "N/A"

    metrics_data = [
        ['Metric', 'Value', 'Interpretation', 'Range'],
        ['Silhouette Score',
         silhouette_val,
         'Higher is better (cluster separation)',
         '[-1, 1]'],
        ['Calinski-Harabasz',
         calinski_val,
         'Higher is better (between/within variance)',
         '[0, ∞]'],
        ['Davies-Bouldin',
         davies_val,
         'LOWER is better (avg similarity ratio)',
         '[0, ∞]'],
    ]

    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.0*inch, 2.5*inch, 1.0*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))

    # Cluster statistics
    story.append(Paragraph("Cluster Statistics", subheading_style))
    stats_data = [
        ['Statistic', 'Count'],
        ['Number of Clusters', str(cluster_quality['n_clusters'])],
        ['Clustered Frames', str(cluster_quality['n_clustered'])],
        ['Noise/Outlier Frames', str(cluster_quality['n_noise'])],
    ]

    stats_table = Table(stats_data, colWidths=[3*inch, 1.5*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))

    # Interpretation
    story.append(Paragraph("Interpretation", subheading_style))

    silhouette = cluster_quality['silhouette']
    davies = cluster_quality['davies_bouldin']

    interpretations = []

    # Silhouette interpretation
    if silhouette is None:
        interpretations.append("⚠ <b>Silhouette Score</b>: Not available (insufficient clusters for calculation).")
    elif silhouette >= 0.5:
        interpretations.append("✅ <b>Silhouette Score</b>: Excellent cluster separation - activities are well-defined.")
    elif silhouette >= 0.3:
        interpretations.append("⚠ <b>Silhouette Score</b>: Moderate cluster separation - some overlap between activities.")
    else:
        interpretations.append("❌ <b>Silhouette Score</b>: Weak cluster separation - activities may be too similar or need different parameters.")

    # Davies-Bouldin interpretation
    if davies is None:
        interpretations.append("⚠ <b>Davies-Bouldin</b>: Not available (insufficient clusters for calculation).")
    elif davies <= 1.0:
        interpretations.append("✅ <b>Davies-Bouldin</b>: Excellent cluster quality - low similarity between different clusters.")
    elif davies <= 1.5:
        interpretations.append("⚠ <b>Davies-Bouldin</b>: Moderate cluster quality - some clusters may be similar.")
    else:
        interpretations.append("❌ <b>Davies-Bouldin</b>: Weak cluster quality - clusters may be poorly separated.")

    # Noise percentage
    noise_pct = cluster_quality['n_noise'] / (cluster_quality['n_clustered'] + cluster_quality['n_noise']) * 100
    if noise_pct <= 10:
        interpretations.append(f"✅ <b>Noise Level</b>: Low ({noise_pct:.1f}%) - most frames fit into clear activity patterns.")
    elif noise_pct <= 25:
        interpretations.append(f"⚠ <b>Noise Level</b>: Moderate ({noise_pct:.1f}%) - some frames don't fit clear patterns.")
    else:
        interpretations.append(f"❌ <b>Noise Level</b>: High ({noise_pct:.1f}%) - many frames are outliers, consider collecting more data or adjusting parameters.")

    for interpretation in interpretations:
        story.append(Paragraph(interpretation, normal_style))
        story.append(Spacer(1, 0.1*inch))

    story.append(Spacer(1, 0.2*inch))

    # Overall quality verdict
    overall_good = (silhouette is not None and silhouette >= 0.3 and
                    davies is not None and davies <= 1.5 and noise_pct <= 25)
    overall_moderate = ((silhouette is not None and silhouette >= 0.2) or
                       (davies is not None and davies <= 2.0 and noise_pct <= 40))

    if overall_good:
        verdict = ("✅ <b>GOOD CLUSTERING QUALITY</b><br/><br/>"
                  "The activity clusters are well-defined and meaningful. This indicates good dataset diversity "
                  "and successful activity differentiation.")
        verdict_color = colors.HexColor('#388e3c')
    elif overall_moderate:
        verdict = ("⚠ <b>MODERATE CLUSTERING QUALITY</b><br/><br/>"
                  "The activity clusters show some structure but could be improved. Consider collecting more "
                  "samples per activity or adjusting clustering parameters.")
        verdict_color = colors.HexColor('#f57c00')
    else:
        verdict = ("❌ <b>WEAK CLUSTERING QUALITY</b><br/><br/>"
                  "The activity clusters are poorly defined. This may indicate insufficient diversity, too few "
                  "samples per activity, or the need for different clustering parameters.")
        verdict_color = colors.HexColor('#d32f2f')

    verdict_style = ParagraphStyle('ClusterVerdict', parent=normal_style, fontSize=11,
                                   textColor=verdict_color, alignment=TA_CENTER,
                                   borderWidth=2, borderColor=verdict_color,
                                   borderPadding=10, backColor=colors.HexColor('#fafafa'))

    story.append(Paragraph(verdict, verdict_style))
    story.append(Spacer(1, 0.2*inch))


def _add_coverage_analysis(story, analysis_data, temp_dir, heading_style, normal_style, image_quality):
    """Page 6+: Coverage Gap Analysis (multi-page if many activities)"""
    story.append(PageBreak())
    story.append(Paragraph("Coverage Gap Analysis", heading_style))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "This heatmap shows the distribution of lighting conditions across different activities. "
        "Gaps (cells with low/zero counts) indicate missing scenarios that may need additional data collection.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    # Check if single heatmap or multiple chunks
    single_heatmap_path = temp_dir / "coverage_heatmap.png"
    chunk_paths = sorted(temp_dir.glob("coverage_heatmap_part*.png"))

    if single_heatmap_path.exists() and not chunk_paths:
        # Single heatmap
        heatmap_chart_path = compress_image_for_pdf(single_heatmap_path, quality=image_quality)
        heatmap_img = PILImage.open(heatmap_chart_path)
        img_width, img_height = heatmap_img.size
        max_width = 6 * inch
        width_scale = max_width / img_width
        scaled_height = min(img_height * width_scale, 8 * inch)
        img = RLImage(str(heatmap_chart_path), width=max_width, height=scaled_height)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
    elif chunk_paths:
        # Multiple heatmap chunks - each on its own section
        for i, chunk_path in enumerate(chunk_paths):
            if i > 0:
                story.append(PageBreak())
            compressed_path = compress_image_for_pdf(chunk_path, quality=image_quality)
            chunk_img = PILImage.open(compressed_path)
            img_width, img_height = chunk_img.size
            max_width = 6 * inch
            width_scale = max_width / img_width
            scaled_height = min(img_height * width_scale, 8 * inch)
            img = RLImage(str(compressed_path), width=max_width, height=scaled_height)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
    else:
        story.append(Paragraph("⚠ Heatmap not generated", normal_style))
        story.append(Spacer(1, 0.3*inch))

    # Final verdict
    story.append(Paragraph("Final Assessment", heading_style))

    stats = analysis_data['summary_stats']
    quality_score = stats['quality_score']

    if quality_score >= 7 and stats['n_activities'] >= 3 and stats['blurry_pct'] < 30:
        verdict = ("✅ <b>READY FOR ANNOTATION</b><br/><br/>"
                  "This dataset shows good quality and diversity. The frames are suitable for annotation effort. "
                  "Consider addressing the recommendations above to further improve the dataset.")
        verdict_color = colors.HexColor('#388e3c')
    elif quality_score >= 5:
        verdict = ("⚠ <b>NEEDS IMPROVEMENT</b><br/><br/>"
                  "This dataset has moderate quality but some issues. Review the recommendations carefully. "
                  "Consider dropping low-quality frames and adding more diverse scenarios before annotation.")
        verdict_color = colors.HexColor('#f57c00')
    else:
        verdict = ("❌ <b>NOT READY FOR ANNOTATION</b><br/><br/>"
                  "This dataset has significant quality issues. Annotation effort on these frames may be wasted. "
                  "Strongly recommend improving frame quality and diversity before proceeding.")
        verdict_color = colors.HexColor('#d32f2f')

    verdict_style = ParagraphStyle('Verdict', parent=normal_style, fontSize=11,
                                   textColor=verdict_color, alignment=TA_CENTER,
                                   borderWidth=2, borderColor=verdict_color,
                                   borderPadding=10, backColor=colors.HexColor('#fafafa'))

    story.append(Paragraph(verdict, verdict_style))

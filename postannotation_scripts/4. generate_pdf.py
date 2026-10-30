"""
PDF Report Generator for Intra-Class Variation Analysis

This script generates a comprehensive PDF report combining:
- Dataset quality statistics from Script 1 (annotation cropping)
- Clustering analysis visualizations from Script 3 (DBSCAN clustering)

Usage:
    python "4. generate_pdf.py" \
        --temp_txt_file temp_ann_file.txt \
        --clustering_dir clustering_results \
        --pdf_name "REPORT" \
        --imgs_path path/to/images \
        --label_path path/to/labels \
        --classes_txt path/to/classes.txt \
        --auto_tune --auto_tune_percentile 95

Output:
    - Page 1: Dataset overview table + pipeline config table + class summary table
    - Pages 2-N: Per-class clustering montages with captions
"""

import os
import argparse
from collections import OrderedDict
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')


def parse_temp_txt_file(txt_path):
    """Parse Script 1 temporary txt file and extract all statistics."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    stats = {
        'raw_content': content,
        'class_data': [],
        'total_annotations': 0,
        'total_crops': 0,
        'imbalance_ratio': None,
        'imbalance_warning': None,
        'total_images': 0,
        'total_labels': 0,
        'background_images': 0,
        'empty_crops': 0,
        'degenerate_crops': 0,
        'empty_annotation_files': 0,
        'invalid_annotation_files': 0,
        'error_reading_files': 0,
        # issue file lists for the detail table
        'missing_img_files': [],           # txt with no matching image
        'unexpected_class_files': [],      # (class_id, count, [files])
        'degenerate_crop_file_list': [],
        'empty_crop_file_list': [],
        'background_image_file_list': [],  # images with no annotation txt
        'empty_annotation_file_list': [],  # txt files that are empty
        'duplicate_basename_count': 0,
        'duplicate_basename_list': [],     # each entry = list of filenames sharing a basename
    }

    in_class_section = False
    for line in content.split('\n'):
        if 'CLASS DISTRIBUTION & SUCCESSFUL CROPS:' in line:
            in_class_section = True
            continue

        if in_class_section:
            if line.startswith('Total annotations in dataset:'):
                stats['total_annotations'] = int(line.split(':')[1].strip())
            elif line.startswith('Total crops successfully extracted:'):
                stats['total_crops'] = int(line.split(':')[1].strip())
            elif line.strip().startswith('Class '):
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    class_info = parts[0].replace('Class ', '').strip()
                    class_id = class_info.split('(')[0].strip()
                    class_name = class_info.split('(')[1].replace(')', '').strip()
                    detail = parts[1].strip()
                    annotations = int(detail.split('annotations')[0].strip())
                    crops_part = detail.split('→')[1].strip()
                    crops = int(crops_part.split('crops')[0].strip())
                    percentage = float(crops_part.split('(')[1].replace('%)', '').strip())
                    stats['class_data'].append({
                        'class_id': class_id,
                        'name': class_name,
                        'annotations': annotations,
                        'crops': crops,
                        'percentage': percentage
                    })
            elif 'CLASS IMBALANCE ANALYSIS:' in line:
                in_class_section = False

        if line.startswith('Imbalance ratio:'):
            ratio_str = line.split(':')[1].strip().replace('x', '')
            stats['imbalance_ratio'] = float(ratio_str)
        elif '⚠️  WARNING:' in line and 'imbalance' in line.lower():
            stats['imbalance_warning'] = line.strip()
        elif '✅ Class distribution is relatively balanced' in line:
            stats['imbalance_warning'] = line.strip()

        # Parse overview numbers
        if line.startswith('Total image files found:'):
            try:
                stats['total_images'] = int(line.split(':')[1].strip())
            except Exception:
                pass
        elif line.startswith('Total annotation files found:'):
            try:
                stats['total_labels'] = int(line.split(':')[1].strip())
            except Exception:
                pass
        elif 'PNG FILES WITH NO TXT ANNOTATIONS' in line:
            pass  # count parsed via section_map below

    # Parse counts from named sections
    section_map = {
        'EMPTY CROP FILES (Zero-size Crops):':          'empty_crops',
        'DEGENERATE CROP FILES':                        'degenerate_crops',
        'EMPTY ANNOTATION FILES (No Objects Labelled):': 'empty_annotation_files',
        'INVALID ANNOTATION FILES (Malformed Data):':   'invalid_annotation_files',
        'ERRORS WHILE READING FILES:':                  'error_reading_files',
        'PNG FILES WITH NO TXT ANNOTATIONS':            'background_images',
        'DUPLICATE IMAGE BASENAMES':                    'duplicate_basename_count',
    }
    current_key = None
    for line in content.split('\n'):
        for header, key in section_map.items():
            if header in line:
                current_key = key
                break
        if current_key and line.startswith('Count:'):
            try:
                stats[current_key] = int(line.split(':')[1].strip())
            except Exception:
                pass
            current_key = None

    # Parse issue file lists for detail table
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        # Background images (PNG with no TXT)
        if 'PNG FILES WITH NO TXT ANNOTATIONS' in line:
            i += 1
            while i < len(lines) and not lines[i].startswith('Count:'):
                i += 1
            i += 1  # skip Count line
            if i < len(lines) and lines[i].strip() == 'Files:':
                i += 1
            while i < len(lines) and lines[i].strip().startswith('- '):
                stats['background_image_file_list'].append(lines[i].strip()[2:])
                i += 1
            continue

        # Empty annotation files (TXT exists but has no objects)
        if 'EMPTY ANNOTATION FILES (No Objects Labelled):' in line:
            i += 1
            while i < len(lines) and not lines[i].startswith('Count:'):
                i += 1
            i += 1  # skip Count line
            if i < len(lines) and lines[i].strip() == 'Files:':
                i += 1
            while i < len(lines) and lines[i].strip().startswith('- '):
                stats['empty_annotation_file_list'].append(lines[i].strip()[2:])
                i += 1
            continue

        # Degenerate crop file list
        if 'DEGENERATE CROP FILES' in line and 'min dimension' in line:
            i += 1
            while i < len(lines) and not lines[i].startswith('Count:'):
                i += 1
            i += 1  # skip Count line
            if i < len(lines) and lines[i].strip() == 'Files:':
                i += 1
            while i < len(lines) and lines[i].strip().startswith('- '):
                stats['degenerate_crop_file_list'].append(lines[i].strip()[2:])
                i += 1
            continue

        # Empty crop file list
        if 'EMPTY CROP FILES (Zero-size Crops):' in line:
            i += 1
            while i < len(lines) and not lines[i].startswith('Count:'):
                i += 1
            i += 1  # skip Count line
            if i < len(lines) and lines[i].strip() == 'Files:':
                i += 1
            while i < len(lines) and lines[i].strip().startswith('- '):
                stats['empty_crop_file_list'].append(lines[i].strip()[2:])
                i += 1
            continue

        # TXT files with no matching image
        if 'TXT FILES WITH NO MATCHING IMAGE ON DISK:' in line:
            i += 1
            while i < len(lines) and not lines[i].startswith('Count:'):
                i += 1
            i += 1
            while i < len(lines) and lines[i].strip().startswith('- '):
                stats['missing_img_files'].append(lines[i].strip()[2:])
                i += 1
            continue

        # Duplicate image basenames
        if 'DUPLICATE IMAGE BASENAMES' in line:
            i += 1
            while i < len(lines) and not lines[i].startswith('Count:'):
                i += 1
            i += 1  # skip Count line
            if i < len(lines) and lines[i].strip() == 'Files:':
                i += 1
            while i < len(lines) and lines[i].strip().startswith('- '):
                stats['duplicate_basename_list'].append(lines[i].strip()[2:])
                i += 1
            continue

        # Unexpected class IDs — collect (class_id, annotation_count, [files])
        if 'UNEXPECTED CLASS IDs IN ANNOTATIONS' in line:
            i += 1
            while i < len(lines) and not lines[i].startswith('Count:'):
                i += 1
            i += 1  # skip Count line
            # skip "Details:" header if present
            if i < len(lines) and lines[i].strip() == 'Details:':
                i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith("Class ID '"):
                    # "Class ID 'N' (M annotations, K file(s)):"
                    try:
                        cls_id = l.split("'")[1]
                        ann_count = int(l.split('(')[1].split(' ')[0])
                        files_in_entry = []
                        i += 1
                        while i < len(lines) and lines[i].strip().startswith('- '):
                            files_in_entry.append(lines[i].strip()[2:])
                            i += 1
                        stats['unexpected_class_files'].append((cls_id, ann_count, files_in_entry))
                    except Exception:
                        i += 1
                elif l == '' or l.startswith('='):
                    break
                else:
                    i += 1
            continue

        i += 1

    return stats


def load_cluster_statistics(clustering_dir):
    """Load cluster statistics CSV from Script 3 output."""
    csv_path = Path(clustering_dir) / "cluster_statistics.csv"
    if not csv_path.exists():
        print(f"⚠️  WARNING: cluster_statistics.csv not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)


def find_montage_images(clustering_dir, class_name):
    """Find all montage image parts (JPG or PNG) for a given class, sorted by part number.

    Returns a list of paths. Single-file montages return a one-element list.
    Multi-part montages (e.g. *_part1.jpg, *_part2.jpg) return all parts in order.
    Returns empty list if nothing found.
    """
    clustering_path = Path(clustering_dir)

    # Check for multi-part files first (e.g. <class>_montage_part1.jpg)
    parts = []
    for ext in ('.jpg', '.png'):
        pattern = f"{class_name}_montage_part*{ext}"
        found = sorted(clustering_path.glob(pattern),
                       key=lambda p: int(''.join(filter(str.isdigit, p.stem.split('_part')[-1])) or '0'))
        if found:
            parts.extend(str(p) for p in found)
    if parts:
        return parts

    # Single file
    for ext in ('.jpg', '.png'):
        montage_path = clustering_path / f"{class_name}_montage{ext}"
        if montage_path.exists():
            return [str(montage_path)]

    # Fuzzy fallback
    for file in sorted(clustering_path.glob("*_montage.*")):
        if file.suffix.lower() in ('.jpg', '.png') and \
                file.stem.lower().replace('_montage', '') == class_name.lower():
            return [str(file)]

    return []


def find_montage_image(clustering_dir, class_name):
    """Legacy single-path wrapper around find_montage_images."""
    results = find_montage_images(clustering_dir, class_name)
    return results[0] if results else None


def create_placeholder_image(class_name, output_path):
    """Create a placeholder image when montage is missing."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f'⚠️  Montage Not Available\n\nClass: {class_name}\n\n'
            f'Possible reasons:\n• Insufficient samples for clustering\n• Clustering failed\n• File missing',
            ha='center', va='center', fontsize=14, color='red',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path


def _make_styles():
    """Return common ReportLab paragraph styles."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
    )
    return title_style, heading_style, normal_style


def _table_style_header(header_color='#1f77b4'):
    """Standard TableStyle for a table with a colored header row."""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ])


def _infer_variant_labels(contrastive_groups):
    """
    Derive legend labels from class name suffixes across all groups.
    Each group has N variants at position 0..N-1. If all groups agree on the
    suffix token at position i (splitting on '_'), use that token; else 'Variant i+1'.
    Returns list of label strings, length = max variants across all groups.
    """
    max_n = max((len(v) for v in contrastive_groups.values()), default=1)
    labels = []
    for i in range(max_n):
        # collect the last token of class name at position i in each group that has one
        tokens = []
        for class_list in contrastive_groups.values():
            if i < len(class_list):
                parts = class_list[i].split('_')
                tokens.append(parts[-1] if parts else class_list[i])
        unique = set(tokens)
        if len(unique) == 1:
            labels.append(tokens[0])
        else:
            labels.append(f'Variant {i + 1}')
    return labels


def _build_contrastive_chart(stats, contrastive_groups, temp_dir):
    """
    Build grouped horizontal bar chart for contrastive groups only (ungrouped classes excluded).
    contrastive_groups: dict[group_name -> list[class_name]]
    Returns path to saved chart PNG, or None on failure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ann_by_class = {d['name']: d['annotations'] for d in stats['class_data']}
    groups = list(contrastive_groups.items())  # dict order preserved
    variant_labels = _infer_variant_labels(contrastive_groups)
    palette = ['#3a7fc1', '#d95f49', '#4caa6b', '#e8a838', '#9b59b6', '#1abc9c']

    fig_h = max(4, len(groups) * 0.6 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_positions = []
    y_labels = []
    bar_height = 0.35
    y = 0
    all_counts = [ann_by_class.get(cls, 0) for _, cls_list in groups for cls in cls_list]
    max_count = max(all_counts) if all_counts else 1

    for group_name, class_list in groups:
        n = len(class_list)
        offsets = [(i - (n - 1) / 2) * bar_height for i in range(n)]
        for cls_idx, (cls_name, offset) in enumerate(zip(class_list, offsets)):
            count = ann_by_class.get(cls_name, 0)
            color = palette[cls_idx % len(palette)]
            ax.barh(y + offset, count, height=bar_height * 0.9,
                    color=color, edgecolor='white', linewidth=0.4)
            ax.text(count + max_count * 0.005, y + offset,
                    f'{count:,}', va='center', ha='left', fontsize=7.5)
        y_positions.append(y)
        y_labels.append(group_name)
        y += 1

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Sample Count', fontsize=10)
    ax.set_title('Contrastive Group Sample Counts', fontsize=12, fontweight='bold')
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # Legend using inferred variant labels
    max_variants = max((len(v) for v in contrastive_groups.values()), default=1)
    if max_variants > 1:
        handles = [Patch(facecolor=palette[i % len(palette)], label=variant_labels[i])
                   for i in range(max_variants)]
        ax.legend(handles=handles, loc='lower right', fontsize=8)

    plt.tight_layout()
    out_path = temp_dir / "contrastive_distribution.png"
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    return out_path


def _parse_n_samples(raw):
    """Parse n_samples field: '4000/15000' → (4000, 15000), '1822' → (1822, 1822)."""
    s = str(raw)
    if '/' in s:
        parts = s.split('/')
        return int(parts[0].strip()), int(parts[1].strip())
    v = int(s.strip())
    return v, v


def _parse_outlier_rate(raw):
    """Parse outlier_rate field: '12.3%' → 12.3."""
    return float(str(raw).replace('%', '').strip())


def _generate_insights_page(cluster_df, stats, story, heading_style, normal_style):
    """Append a condensed Insights & Recommendations page — one row per issue type, classes listed inside."""
    if cluster_df is None or cluster_df.empty:
        return

    RED    = colors.HexColor('#c0392b')
    ORANGE = colors.HexColor('#d35400')
    GREEN  = colors.HexColor('#27ae60')
    BLUE   = colors.HexColor('#2980b9')

    # issue_key → (severity, flag_text, recommendation_text, [class_names])
    # Ordered dict preserves insertion order = rule priority order
    issue_map = OrderedDict()

    def _add(key, sev, flag, rec, cls):
        if key not in issue_map:
            issue_map[key] = (sev, flag, rec, [])
        issue_map[key][3].append(cls)

    for _, row in cluster_df.iterrows():
        cls = str(row['class'])
        n_clustered, n_total = _parse_n_samples(row['n_samples'])
        n_clusters   = int(row['n_clusters'])
        outlier_rate = _parse_outlier_rate(row['outlier_rate'])
        eps_used     = float(row['eps_used'])
        downsampled  = str(row.get('downsampled', 'no')).lower() == 'yes'

        if outlier_rate > 8:
            _add('high_outlier', 'critical', 'High outlier rate',
                 'Review crops for annotation errors, truncated bboxes, or wrong-class labels', cls)
        elif outlier_rate > 3:
            _add('mod_outlier', 'warning', 'Moderate outlier rate',
                 'Spot-check outlier folder; may be acceptable if class is visually diverse', cls)

        if n_clusters > 15:
            _add('high_variation', 'critical', 'High intra-class variation',
                 'Verify class definition is consistent; consider splitting into sub-classes', cls)
        elif n_clusters > 5:
            _add('mod_clusters', 'warning', 'Moderate cluster count',
                 'Review cluster montages — may reflect real sub-variants or lighting differences', cls)

        if n_clusters == 0:
            _add('all_noise', 'critical', 'All samples noise',
                 'DBSCAN found no coherent structure — loosen eps or inspect data quality', cls)

        if n_clusters == 1 and outlier_rate > 8:
            _add('tight_core_noise', 'warning', 'Tight core + scattered noise',
                 'Possible bimodal class; inspect outlier folder — may be unlabelled variant', cls)

        if n_clustered < 50:
            _add('insufficient_data', 'critical', 'Insufficient data',
                 'Results unreliable — collect more samples before drawing conclusions', cls)
        elif n_clustered < 200:
            _add('low_samples', 'warning', 'Low sample count',
                 'Model may underfit this class; collect more if possible', cls)

        if downsampled and outlier_rate > 8:
            _add('sys_noise', 'warning', 'Systematic annotation noise',
                 'Visually homogeneous but high outliers — likely labelling errors, not diversity', cls)

        if eps_used > 0.5:
            _add('spread', 'warning', 'Embeddings very spread',
                 'Class may contain fundamentally different visual concepts — review montage', cls)

        if eps_used < 0.05 and n_clusters > 5:
            _add('over_frag', 'warning', 'Over-fragmented at tight eps',
                 'Many micro-clusters; may be noise-sensitive — try a looser eps (raise the percentile, or set eps manually)', cls)

        if n_clusters > 3 and n_clustered < 300:
            _add('over_cluster_small', 'warning', 'Over-clustering on small class',
                 'Too many clusters for sample count — likely noise; raise min_cluster_size', cls)

        if n_clusters <= 2 and n_clustered > 1000 and outlier_rate <= 3:
            _add('homogeneous', 'ok', 'Visually homogeneous',
                 'Class is consistent — good for training', cls)

        if outlier_rate == 0.0 and n_clustered > 500:
            _add('no_outliers', 'info', 'No outliers detected',
                 'Either very clean data or eps too loose — verify montage looks homogeneous', cls)

        if downsampled:
            _add('downsampled', 'info', 'Large class downsampled',
                 'Clustering ran on a sample — results representative but not exhaustive', cls)

    story.append(Paragraph("INSIGHTS & RECOMMENDATIONS", heading_style))
    story.append(Paragraph(
        "Grouped by issue type. Each row lists all classes that triggered the flag. "
        "Derived from embedding structure — use as a starting point for review.",
        normal_style))
    story.append(Spacer(1, 0.12 * inch))

    if not issue_map:
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph(
            "✅ All classes passed all checks. Dataset looks clean.",
            ParagraphStyle('GoodNews', parent=normal_style, fontSize=11,
                           textColor=GREEN, fontName='Helvetica-Bold')))
        return

    SEV_ORDER = ['critical', 'warning', 'ok', 'info']
    SEV_HEADER = {
        'critical': ('🔴 Critical Issues', RED),
        'warning':  ('🟡 Warnings',        ORANGE),
        'ok':       ('🟢 OK',              GREEN),
        'info':     ('ℹ Info',             BLUE),
    }

    # Group issues by severity
    grouped = {s: [] for s in SEV_ORDER}
    for key, (sev, flag, rec, classes) in issue_map.items():
        grouped[sev].append((flag, rec, classes))

    col_widths = [1.8*inch, 2.5*inch, 3.2*inch]

    for sev in SEV_ORDER:
        entries = grouped[sev]
        if not entries:
            continue

        header_text, header_color = SEV_HEADER[sev]
        story.append(Paragraph(header_text, ParagraphStyle(
            f'SevHead_{sev}', parent=normal_style, fontSize=10,
            fontName='Helvetica-Bold', textColor=header_color,
            spaceBefore=8, spaceAfter=3)))

        table_rows = [['Issue', 'Recommendation', 'Classes Affected']]
        for flag, rec, classes in entries:
            classes_str = ', '.join(classes)
            table_rows.append([
                Paragraph(flag, ParagraphStyle('IF', parent=normal_style, fontSize=8, fontName='Helvetica-Bold')),
                Paragraph(rec,  ParagraphStyle('IR', parent=normal_style, fontSize=8, leading=11)),
                Paragraph(classes_str, ParagraphStyle('IC', parent=normal_style, fontSize=7.5, leading=10,
                                                       textColor=colors.HexColor('#333333'))),
            ])

        tbl = Table(table_rows, colWidths=col_widths, repeatRows=1)
        ts = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8.5),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ])
        tbl.setStyle(ts)
        story.append(tbl)
        story.append(Spacer(1, 0.08*inch))


def build_pdf(temp_txt_file, clustering_dir, pdf_name, pdf_quality,
              imgs_path=None, label_path=None, classes_txt=None,
              auto_tune=False, auto_tune_percentile=95, epsilon=0.15,
              contrastive_groups=None, eps_method="knee"):
    """Build the complete PDF report."""
    print("\n" + "="*70)
    print("PDF REPORT GENERATION")
    print("="*70)

    if not os.path.isfile(temp_txt_file):
        raise FileNotFoundError(f"❌ Temp txt file not found: {temp_txt_file}")
    if not os.path.isdir(clustering_dir):
        raise NotADirectoryError(f"❌ Clustering directory not found: {clustering_dir}")

    print("\n📊 Parsing Script 1 statistics...")
    stats = parse_temp_txt_file(temp_txt_file)
    print(f"   ✅ Found {len(stats['class_data'])} classes")
    print(f"   ✅ Total annotations: {stats['total_annotations']}")

    print("\n📊 Loading Script 3 cluster statistics...")
    cluster_df = load_cluster_statistics(clustering_dir)
    if cluster_df is not None:
        print(f"   ✅ Loaded statistics for {len(cluster_df)} classes")
    else:
        print("   ⚠️  No cluster statistics found")

    pdf_filename = f"{pdf_name}.pdf" if not pdf_name.endswith('.pdf') else pdf_name
    temp_dir = Path(clustering_dir).resolve() / "temp_pdf_charts"
    temp_dir.mkdir(exist_ok=True)
    print(f"\n📄 Creating PDF: {pdf_filename}")

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch,
                            leftMargin=0.5*inch, rightMargin=0.5*inch)

    title_style, heading_style, normal_style = _make_styles()
    story = []

    # =========================================================================
    # PAGE 1: Professional overview tables
    # =========================================================================
    story.append(Paragraph("INTRA-CLASS VARIATION ANALYSIS", title_style))
    story.append(Paragraph("Dataset Quality Report", ParagraphStyle(
        'SubTitle', parent=heading_style, fontSize=12, alignment=TA_CENTER, spaceBefore=2)))
    story.append(Spacer(1, 0.25*inch))

    # --- Section: SCRIPT 1 — Dataset Overview ---
    story.append(Paragraph("SCRIPT 1: ANNOTATION BBOX CROPPING — DATASET OVERVIEW", heading_style))

    n_classes = len(stats['class_data'])
    imbalance_ratio = stats.get('imbalance_ratio')
    if imbalance_ratio is not None:
        if imbalance_ratio > 10:
            imbalance_str = f"{imbalance_ratio:.1f}x  ⚠ Severe"
        elif imbalance_ratio > 5:
            imbalance_str = f"{imbalance_ratio:.1f}x  ⚠ Moderate"
        else:
            imbalance_str = f"{imbalance_ratio:.1f}x  ✅ Balanced"
    else:
        imbalance_str = "N/A"

    dataset_rows = [
        ['Metric', 'Value'],
    ]
    if imgs_path:
        dataset_rows.append(['Images Folder', str(imgs_path)])
    if label_path:
        dataset_rows.append(['Labels Folder', str(label_path)])
    if classes_txt:
        dataset_rows.append(['Classes File', str(classes_txt)])
    dataset_rows += [
        ['Total Image Files Found', str(stats.get('total_images', 'N/A'))],
        ['Total Annotation Files Found', str(stats.get('total_labels', 'N/A'))],
        ['Background Images (no annotations)', str(stats.get('background_images', 'N/A'))],
        ['Number of Classes', str(n_classes)],
        ['Total Annotations in Dataset', str(stats['total_annotations'])],
        ['Total Crops Extracted', str(stats['total_crops'])],
        ['Class Imbalance Ratio (max/min)', imbalance_str],
    ]

    bad_crops = stats.get('empty_crops', 0) + stats.get('degenerate_crops', 0)
    bad_crops_str = f"{bad_crops}  (empty: {stats.get('empty_crops', 0)}, tiny <3px: {stats.get('degenerate_crops', 0)})"
    if bad_crops > 0:
        bad_crops_str = f"⚠  {bad_crops_str}"
    dataset_rows.append(['Bad Crops (skipped)', bad_crops_str])

    def _warn(n, label):
        return f"⚠  {n} {label}" if n > 0 else str(n)

    dataset_rows.append(['Empty Annotation Files (no objects)', _warn(stats.get('empty_annotation_files', 0), 'file(s)')])
    dataset_rows.append(['Invalid Annotation Files (malformed)', _warn(stats.get('invalid_annotation_files', 0), 'file(s)')])
    dataset_rows.append(['Errors Reading Image Files', _warn(stats.get('error_reading_files', 0), 'file(s)')])
    dup_count = stats.get('duplicate_basename_count', 0)
    dataset_rows.append(['Duplicate Image Basenames (multi-format)', _warn(dup_count, 'basename(s) — same file in .png + .jpg')])

    dataset_table = Table(dataset_rows, colWidths=[3*inch, 4.5*inch])
    dataset_table.setStyle(_table_style_header('#2c3e50'))
    story.append(dataset_table)
    story.append(Spacer(1, 0.2*inch))

    # --- Section: Issue Files Detail (only rendered if any issues exist) ---
    issue_rows = []

    def _add_issue_block(rows, header, file_list):
        if not file_list:
            return
        rows.append([Paragraph(f"<b>{header}</b>", ParagraphStyle(
            'IssueHeader', parent=normal_style, fontSize=9,
            textColor=colors.HexColor('#c0392b'), fontName='Helvetica-Bold')), ''])
        for entry in file_list:
            rows.append(['', Paragraph(entry, ParagraphStyle(
                'IssueFile', parent=normal_style, fontSize=8,
                fontName='Helvetica', textColor=colors.HexColor('#555555')))])

    _add_issue_block(issue_rows, "Background Images (no annotation TXT)", stats['background_image_file_list'])
    _add_issue_block(issue_rows, "Empty Annotation Files (no objects)", stats['empty_annotation_file_list'])
    _add_issue_block(issue_rows, "Degenerate Crops (<3px)", stats['degenerate_crop_file_list'])
    _add_issue_block(issue_rows, "Empty Crops (zero-size)", stats['empty_crop_file_list'])
    _add_issue_block(issue_rows, "TXT Files — Image Not Found", stats['missing_img_files'])
    _add_issue_block(issue_rows, "Duplicate Image Basenames (multi-format)", stats['duplicate_basename_list'])

    for cls_id, ann_count, files in stats['unexpected_class_files']:
        _add_issue_block(
            issue_rows,
            f"Unexpected Class ID '{cls_id}' ({ann_count} annotations — labelling ignored)",
            files
        )

    if issue_rows:
        story.append(Paragraph("ANNOTATION ISSUE FILES", heading_style))
        issue_table = Table(
            [['Issue Type', 'File']] + issue_rows,
            colWidths=[2.5*inch, 5*inch]
        )
        issue_ts = _table_style_header('#922b21')
        issue_table.setStyle(issue_ts)
        story.append(issue_table)
        story.append(Spacer(1, 0.25*inch))

    # --- Section: Pipeline Configuration ---
    story.append(Paragraph("PIPELINE CONFIGURATION", heading_style))

    if auto_tune and eps_method == "knee":
        eps_config_label = "Auto-Tune Eps Rule"
        eps_config_value = "knee of sorted k-distance curve"
    elif auto_tune:
        eps_config_label = "Auto-Tune k-NN Percentile (caps outliers near 100-p%)"
        eps_config_value = str(auto_tune_percentile)
    else:
        eps_config_label = "DBSCAN Epsilon (manual)"
        eps_config_value = str(epsilon)

    config_rows = [
        ['Parameter', 'Value'],
        ['Embedding Model', 'facebook/dinov2-base (768d)'],
        ['Auto-Tune Eps', 'Yes' if auto_tune else 'No'],
        [eps_config_label, eps_config_value],
    ]

    config_table = Table(config_rows, colWidths=[3*inch, 4.5*inch])
    config_table.setStyle(_table_style_header('#1f77b4'))
    story.append(config_table)
    story.append(Spacer(1, 0.25*inch))

    # --- Section: Annotation Distribution Bar Chart ---
    if stats['class_data']:
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        def _append_chart_to_story(chart_path, story, max_h=6.5 * inch):
            if chart_path and chart_path.exists():
                img = PILImage.open(chart_path)
                bw, bh = img.size
                img.close()
                max_w = 7 * inch
                scaled_h = bh * (max_w / bw)
                if scaled_h > max_h:
                    scale = max_h / scaled_h
                    max_w *= scale
                    scaled_h = max_h
                story.append(Image(str(chart_path), width=max_w, height=scaled_h))
                story.append(Spacer(1, 0.2 * inch))

        # Plain annotation distribution — always shown
        bar_chart_path = temp_dir / "annotation_distribution.png"
        class_names_bar = [d['name'] for d in stats['class_data']]
        ann_counts = [d['annotations'] for d in stats['class_data']]

        fig_w = max(10, len(class_names_bar) * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, 4.5))
        bars = ax.bar(range(len(class_names_bar)), ann_counts, color='#3a7fc1', edgecolor='#2c5f8a', linewidth=0.5)

        for bar, count in zip(bars, ann_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ann_counts) * 0.005,
                    str(count), ha='center', va='bottom', fontsize=7, rotation=0)

        ax.set_xticks(range(len(class_names_bar)))
        ax.set_xticklabels(class_names_bar, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Annotation Count', fontsize=10)
        ax.set_xlabel('Class', fontsize=10)
        ax.set_title('Annotation Distribution per Class', fontsize=12, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(bar_chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        _append_chart_to_story(bar_chart_path, story)

        # Contrastive group chart — only when groups provided, shown after plain chart
        if contrastive_groups:
            contrastive_chart_path = _build_contrastive_chart(stats, contrastive_groups, temp_dir)
            _append_chart_to_story(contrastive_chart_path, story)

    # --- Section: Per-Class Summary (replaces imbalance graph) ---
    story.append(Paragraph("CLASS DISTRIBUTION & CLUSTERING SUMMARY", heading_style))

    class_summary_rows = [['Class', 'Samples', 'Clusters', 'Outliers', 'Outlier %', 'Eps Used', 'Downsampled']]

    for class_info in stats['class_data']:
        class_name = class_info['name']
        samples = str(class_info['crops'])
        clusters = 'N/A'
        outliers = 'N/A'
        outlier_pct = 'N/A'
        eps_used = 'N/A'
        downsampled = 'N/A'

        if cluster_df is not None:
            row = cluster_df[cluster_df['class'] == class_name]
            if not row.empty:
                r = row.iloc[0]
                clusters = str(r['n_clusters'])
                outliers = str(r['n_outliers'])
                outlier_pct = str(r['outlier_rate'])
                eps_used = f"{r['eps_used']:.4f}"
                downsampled = 'Yes' if ('downsampled' in r and r['downsampled'] == 'yes') else 'No'

        class_summary_rows.append([class_name, samples, clusters, outliers, outlier_pct, eps_used, downsampled])

    col_widths = [1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.75*inch, 0.85*inch, 0.95*inch]
    class_table = Table(class_summary_rows, colWidths=col_widths)
    ts = _table_style_header('#34495e')
    ts.add('ALIGN', (1, 0), (-1, -1), 'CENTER')
    class_table.setStyle(ts)
    story.append(class_table)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2+: Centroid overview plots (one page per centroid_overview_N.png)
    # =========================================================================
    centroid_pages = sorted(Path(clustering_dir).glob("centroid_overview_*.png"))
    for cp in centroid_pages:
        page_num = cp.stem.replace("centroid_overview_", "")
        story.append(Paragraph(f"Cluster Centroid Overview — Page {page_num}", heading_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Each point = cluster centroid (●) or sampled outlier (✕), colored by class. "
            "Single UMAP fit on centroids + outliers — fast overview of inter-class and intra-class structure.",
            normal_style))
        story.append(Spacer(1, 0.15*inch))
        story.append(Image(str(cp), width=7*inch, height=5.5*inch))
        story.append(PageBreak())

    # =========================================================================
    # PAGES N+: Per-Class Clustering Montages
    # =========================================================================
    print("\n🖼️  Adding class montages...")
    missing_montages = []

    for class_info in stats['class_data']:
        class_name = class_info['name']

        montage_paths = find_montage_images(clustering_dir, class_name)
        if not montage_paths:
            print(f"   ⚠️  WARNING: Montage missing for class '{class_name}' - creating placeholder")
            missing_montages.append(class_name)
            placeholder_path = temp_dir / f"{class_name}_placeholder.png"
            create_placeholder_image(class_name, placeholder_path)
            montage_paths = [str(placeholder_path)]
        else:
            label = f"({len(montage_paths)} parts)" if len(montage_paths) > 1 else ""
            print(f"   ✅ Found montage for class '{class_name}' {label}".strip())

        cluster_info = None
        if cluster_df is not None:
            cluster_row = cluster_df[cluster_df['class'] == class_name]
            if not cluster_row.empty:
                cluster_info = cluster_row.iloc[0]

        story.append(Paragraph(f"Class: {class_name}", ParagraphStyle(
            'ClassTitle', parent=title_style, fontSize=16)))
        story.append(Spacer(1, 0.1*inch))

        try:
            from PIL import Image as PILImage
            PILImage.MAX_IMAGE_PIXELS = None

            max_width = 7 * inch
            max_page_height = 8.5 * inch

            for part_idx, montage_path in enumerate(montage_paths):
                img_pil = PILImage.open(montage_path)
                img_width, img_height = img_pil.size
                width_scale = max_width / img_width
                scaled_height = img_height * width_scale

                if part_idx > 0:
                    story.append(Spacer(1, 0.05*inch))
                    story.append(Paragraph(
                        f"<i>(Part {part_idx + 1} of {len(montage_paths)})</i>",
                        ParagraphStyle('PartLabel', parent=normal_style,
                                       fontSize=9, alignment=TA_CENTER,
                                       textColor=colors.grey)))
                    story.append(Spacer(1, 0.05*inch))

                if scaled_height <= max_page_height:
                    compressed_path = temp_dir / f"{class_name}_part{part_idx}_compressed.jpg"
                    img_pil.convert('RGB').save(compressed_path, 'JPEG', quality=pdf_quality, optimize=True)
                    img_pil.close()
                    story.append(Image(str(compressed_path), width=max_width, height=scaled_height))
                else:
                    if cluster_info is not None:
                        num_cluster_rows = int(cluster_info['n_clusters']) + (1 if cluster_info['n_outliers'] > 0 else 0)
                        # Distribute rows evenly across parts
                        rows_this_part = max(1, num_cluster_rows // len(montage_paths))
                    else:
                        rows_this_part = max(1, int(img_height / 600))

                    row_height_px = img_height / rows_this_part
                    row_height_scaled = row_height_px * width_scale
                    rows_per_page = max(1, int(max_page_height / row_height_scaled))
                    num_chunks = (rows_this_part + rows_per_page - 1) // rows_per_page

                    for chunk_idx in range(num_chunks):
                        start_row = chunk_idx * rows_per_page
                        end_row = min((chunk_idx + 1) * rows_per_page, rows_this_part)
                        top = int(start_row * row_height_px)
                        bottom = int(end_row * row_height_px)
                        chunk = img_pil.crop((0, top, img_width, bottom))
                        chunk_path = temp_dir / f"{class_name}_part{part_idx}_chunk_{chunk_idx}.jpg"
                        chunk.convert('RGB').save(chunk_path, 'JPEG', quality=pdf_quality, optimize=True)
                        chunk_height = (bottom - top) * width_scale

                        if chunk_idx > 0:
                            story.append(Spacer(1, 0.05*inch))
                            story.append(Paragraph(
                                f"<i>(Continuation — Clusters {start_row} to {end_row-1})</i>",
                                ParagraphStyle('Continuation', parent=normal_style,
                                               fontSize=9, alignment=TA_CENTER,
                                               textColor=colors.grey)))
                            story.append(Spacer(1, 0.05*inch))

                        story.append(Image(str(chunk_path), width=max_width, height=chunk_height))
                        if chunk_idx < num_chunks - 1:
                            story.append(PageBreak())
                    img_pil.close()

                if part_idx < len(montage_paths) - 1:
                    story.append(PageBreak())

        except Exception as e:
            import traceback
            print(f"   ⚠️  Error loading image for {class_name}: {e}")
            print(f"   ⚠️  Full error:\n{traceback.format_exc()}")
            story.append(Paragraph("⚠ Error loading montage image", normal_style))

        story.append(Spacer(1, 0.1*inch))

        # Caption
        if cluster_info is not None:
            downsampled_info = ""
            if 'downsampled' in cluster_info and cluster_info['downsampled'] == 'yes':
                downsampled_info = " | <b><font color='orange'>⚠ Downsampled (uniform class)</font></b>"
            caption = (f"<b>Samples:</b> {cluster_info['n_samples']} | "
                       f"<b>Clusters:</b> {cluster_info['n_clusters']} | "
                       f"<b>Outliers:</b> {cluster_info['n_outliers']} ({cluster_info['outlier_rate']}) | "
                       f"<b>Epsilon:</b> {cluster_info['eps_used']:.4f}{downsampled_info}")
        else:
            caption = f"<b>Samples:</b> {class_info['crops']} | <b>Clustering statistics unavailable</b>"

        story.append(Paragraph(caption, ParagraphStyle(
            'Caption', parent=normal_style, fontSize=11, alignment=TA_CENTER)))
        story.append(PageBreak())

    # =========================================================================
    # LAST PAGE: Insights & Recommendations
    # =========================================================================
    if cluster_df is not None:
        story.append(PageBreak())
        _generate_insights_page(cluster_df, stats, story, heading_style, normal_style)

    # Build PDF
    print("\n📦 Building PDF document...")
    doc.build(story)

    print("🧹 Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)

    print("\n" + "="*70)
    print("✅ PDF GENERATION COMPLETE")
    print("="*70)
    print(f"📄 Output: {pdf_filename}")
    print(f"📊 Total pages: ~{len(stats['class_data']) + 2}")

    if missing_montages:
        print(f"\n⚠️  WARNING: {len(missing_montages)} class(es) had missing montages:")
        for cls in missing_montages:
            print(f"   - {cls}")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate PDF report from clustering analysis")
    parser.add_argument("--temp_txt_file", required=True)
    parser.add_argument("--clustering_dir", required=True)
    parser.add_argument("--pdf_name", required=True)
    parser.add_argument("--imgs_path", default=None)
    parser.add_argument("--label_path", default=None)
    parser.add_argument("--classes_txt", default=None)
    parser.add_argument("--auto_tune", action="store_true")
    parser.add_argument("--auto_tune_percentile", type=int, default=95)
    parser.add_argument("--eps_method", choices=["knee", "percentile"], default="knee")
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--pdf_quality", type=int, default=75,
                        help="JPEG compression quality for images in PDF (1-95, default 75)")
    parser.add_argument("--contrastive_groups_json", default=None,
                        help="JSON string: {group_name: [class_a, class_b]} for grouped distribution chart")

    args = parser.parse_args()

    contrastive_groups = None
    if args.contrastive_groups_json:
        import json
        try:
            contrastive_groups = json.loads(args.contrastive_groups_json)
        except Exception as e:
            print(f"⚠️  Could not parse --contrastive_groups_json: {e} — using plain chart")

    build_pdf(
        temp_txt_file=args.temp_txt_file,
        clustering_dir=args.clustering_dir,
        pdf_name=args.pdf_name,
        pdf_quality=args.pdf_quality,
        imgs_path=args.imgs_path,
        label_path=args.label_path,
        classes_txt=args.classes_txt,
        auto_tune=args.auto_tune,
        auto_tune_percentile=args.auto_tune_percentile,
        eps_method=args.eps_method,
        epsilon=args.epsilon,
        contrastive_groups=contrastive_groups,
    )


if __name__ == "__main__":
    main()

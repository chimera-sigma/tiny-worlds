#!/usr/bin/env python3
"""Create placeholder figure images for LaTeX compilation."""

# Try to create simple placeholder images
try:
    from PIL import Image, ImageDraw, ImageFont
    
    figures = [
        ('fig_e3_capacity_rnn_gru.png', 'Figure 1: Capacity vs ΔNLL\n(RNN and GRU)'),
        ('fig_e3_geometry_vs_dynamics_rnn.png', 'Figure 2: Geometry vs Dynamics\n(E3 variants, RNN-16)'),
        ('fig_e2_regime_decoding.png', 'Figure 3: E2 Regime Decoding\n(Struct vs Null, RNN vs GRU)')
    ]
    
    for filename, text in figures:
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw placeholder text
        draw.rectangle([10, 10, 790, 590], outline='black', width=2)
        draw.text((400, 280), text, fill='black', anchor='mm')
        draw.text((400, 320), '[Install matplotlib to generate]', fill='gray', anchor='mm')
        
        img.save(filename)
        print(f"Created placeholder: {filename}")
    
    print("\nPlaceholders created successfully!")
    print("Run 'pip install matplotlib && python3 ../scripts/analyze_results.py' to generate real figures.")

except ImportError:
    print("PIL not available. Creating empty placeholder files...")
    import os
    
    figures = [
        'fig_e3_capacity_rnn_gru.png',
        'fig_e3_geometry_vs_dynamics_rnn.png',
        'fig_e2_regime_decoding.png'
    ]
    
    for filename in figures:
        # Create empty PNG file
        with open(filename, 'wb') as f:
            # Minimal valid PNG header
            f.write(b'\x89PNG\r\n\x1a\n')
        print(f"Created empty placeholder: {filename}")
    
    print("\nEmpty placeholders created. LaTeX will show missing images.")
    print("Install matplotlib: pip install matplotlib")
    print("Then run: python3 ../scripts/analyze_results.py")

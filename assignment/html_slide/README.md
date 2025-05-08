# HTML to PDF Presentation Template (16:9)

A simple HTML-based presentation template designed for PDF conversion with standard 16:9 presentation aspect ratio.

## Features

- Clean, modern design optimized for PDF output
- Standard 16:9 aspect ratio for slides (same as PowerPoint/Keynote)
- Each slide set to print on a separate page
- Proper page breaks for PDF conversion
- Multiple slide layout options (title, two columns, etc.)
- Optimized typography for high-quality output

## How to Use

1. **Edit the content**:
   - Modify the HTML in `index.html` to change the content of your slides
   - Each slide is contained in a `<div class="slide">` element
   - Add more slides by duplicating these elements and incrementing the ID

2. **Customize the appearance**:
   - Edit `css/pdf-styles.css` to change colors, fonts, and layout
   - The template uses a blue and dark gray color scheme by default

3. **Converting to PDF**:
   - Open the `index.html` file in a modern web browser (Chrome or Firefox recommended)
   - Use the browser's print function (Ctrl+P or Cmd+P)
   - Set the destination to "Save as PDF"
   - Make sure "Background graphics" is checked in the print options
   - For best results, select "Landscape" orientation to match the 16:9 ratio
   - Click "Save" to generate your PDF

## PDF Printing Tips

For the best PDF output:
- Use **Landscape orientation** for proper 16:9 aspect ratio
- Enable "Background Colors and Images" in print settings
- Set margins to "None" or "Minimum"
- Disable headers and footers
- Select "Scale: 100%" for accurate sizing
- Preview the PDF before saving to ensure proper page breaks

## Adding Slides

To add a new slide, copy an existing slide div and update its ID:

```html
<div class="slide" id="slide6">
    <div class="slide-content">
        <h2>New Slide Title</h2>
        <p>New slide content goes here.</p>
    </div>
</div>
```

## Slide Layouts

The template includes several example layouts:

1. **Title Slide** (slide1)
2. **Bullet Points** (slide2)
3. **Content with Image** (slide3)
4. **Two Column Layout** (slide4)
5. **Thank You/End Slide** (slide5)

Feel free to mix and match these elements to create your own layouts.

## Browser Compatibility

This template works best in:
- Chrome
- Firefox
- Edge

Note: Safari may have some limitations with print-to-PDF page breaks.

## License

Free to use for personal and commercial projects. 

// Downloads AI-generated (non-real) faces from thispersondoesnotexist.com
// to build a training dataset. Run with: npm install && node download_faces.js

import fetch from "node-fetch";
import fs from "fs";
import path from "path";

// Images are saved as person_<START_INDEX + 1>.jpg ... person_<NUM_IMAGES>.jpg
const START_INDEX = 0;
const NUM_IMAGES = 1000;
const OUTPUT_DIR = path.resolve("./dataset2");

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Function to download an image
async function downloadImage(url, filename) {
    try {
        const response = await fetch(url);

        // Validate content type to ensure it's an image
        const contentType = response.headers.get("content-type");
        if (!contentType || !contentType.startsWith("image")) {
            throw new Error(`Unexpected content type: ${contentType}`);
        }

        // Save image
        const arrayBuffer = await response.arrayBuffer();
        fs.writeFileSync(filename, Buffer.from(arrayBuffer));
        console.log(`Downloaded: ${filename}`);
    } catch (error) {
        console.error(`Failed to download image: ${error.message}`);
    }
}

// Function to retry fetching images
async function fetchImages(url, numImages) {
    let downloaded = START_INDEX;

    while (downloaded < numImages) {
        const filename = path.join(OUTPUT_DIR, `person_${downloaded + 1}.jpg`);
        await downloadImage(url, filename);

        downloaded++;
        // Optional delay to avoid overwhelming the server
        await new Promise((resolve) => setTimeout(resolve, 1000)); // 1 second delay
    }
}

// Run the script
const IMAGE_URL = "https://thispersondoesnotexist.com";
fetchImages(IMAGE_URL, NUM_IMAGES).catch(console.error);

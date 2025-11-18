import { defineConfig } from "vite";

export default defineConfig({
    base: './',
    server: {
        headers: {
            // Это важно!
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
        },
    },
    build: {
        target: "esnext",
        assetsDir: './',
        assetsInlineLimit: 0,
    },
});
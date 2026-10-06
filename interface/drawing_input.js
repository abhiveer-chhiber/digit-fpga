function boxAxis(source, size) {
    const scale = source / size, support = Math.max(1, scale);
    return Array.from({ length: size }, (_, i) => {
        const center = (i + 0.5) * scale;
        const indices = [];
        for (let j = Math.max(0, Math.trunc(center - support / 2 + 0.5));
            j < Math.min(source, Math.trunc(center + support / 2 + 0.5)); j++) {
            const distance = (j - center + 0.5) / support;
            if (distance > -0.5 && distance <= 0.5) indices.push(j);
        }
        return {
            indices,
            weight: Math.floor(4194304 / indices.length + 0.5),
        };
    });
}

function average(bytes, offset, stride, filter) {
    const sum = filter.indices.reduce(
        (total, i) => total + bytes[offset + i * stride], 0
    );
    return Math.min(
        255, Math.floor((sum * filter.weight + 2097152) / 4194304)
    );
}

export function preprocessInk(ink, width, height, resolution = 16) {
    if (ink.length !== width * height) {
        throw new Error("Invalid image dimensions");
    }

    let left = width, top = height, right = 0, bottom = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (ink[y * width + x] <= 32) continue;
            left = Math.min(left, x);
            top = Math.min(top, y);
            right = Math.max(right, x + 1);
            bottom = Math.max(bottom, y + 1);
        }
    }

    if (right === 0) throw new Error("Draw a digit first.");

    const cropWidth = right - left, cropHeight = bottom - top;
    const side = Math.floor(
        Math.max(cropWidth, cropHeight) / 0.75 + 0.5
    );
    const square = new Uint8Array(side * side);
    const offsetX = Math.floor((side - cropWidth) / 2);
    const offsetY = Math.floor((side - cropHeight) / 2);

    for (let y = 0; y < cropHeight; y++) {
        for (let x = 0; x < cropWidth; x++) {
            square[(offsetY + y) * side + offsetX + x] =
                ink[(top + y) * width + left + x];
        }
    }

    let output = square;
    if (side !== resolution) {
        const filters = boxAxis(side, resolution);
        const horizontal = new Uint8Array(side * resolution);

        for (let y = 0; y < side; y++) {
            for (let x = 0; x < resolution; x++) {
                horizontal[y * resolution + x] =
                    average(square, y * side, 1, filters[x]);
            }
        }

        output = new Uint8Array(resolution * resolution);
        for (let y = 0; y < resolution; y++) {
            for (let x = 0; x < resolution; x++) {
                output[y * resolution + x] =
                    average(horizontal, x, resolution, filters[y]);
            }
        }
    }

    return {
        pixels: Array.from(output, value => Math.fround(value / 255)),
        bounds: [left, top, right, bottom],
        edgeTouching:
            left === 0 || top === 0 || right === width || bottom === height,
    };
}

export function preprocessDrawing(image, resolution = 16) {
    const ink = new Uint8Array(image.width * image.height);
    for (let i = 0; i < ink.length; i++) {
        const j = i * 4, alpha = image.data[j + 3] / 255;
        const gray = Math.floor(
            (19595 * image.data[j] + 38470 * image.data[j + 1]
                + 7471 * image.data[j + 2] + 32768) / 65536
        );
        ink[i] = 255 - Math.round(gray * alpha + 255 * (1 - alpha));
    }
    return preprocessInk(ink, image.width, image.height, resolution);
}

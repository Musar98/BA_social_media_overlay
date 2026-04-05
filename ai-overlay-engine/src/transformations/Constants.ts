/**
 * Number of tone-curve segments expected by the shader.
 * This is fixed here because the fragment shader declares u_toneCurve[8].
 */
export const TONE_CURVE_STEPS = 8;

/**
 * Number of color-curve segments expected by the shader.
 * This is fixed here because the fragment shader declares u_colorCurve[8].
 */
export const COLOR_CURVE_STEPS = 8;

/**
 * Maximum number of pending GPU timer queries we keep alive at once.
 *
 * Why this exists:
 * GPU timer queries resolve asynchronously. A small cap prevents unbounded
 * growth if frames are rendered faster than results become available.
 */
export const MAX_PENDING_GPU_QUERIES = 8;

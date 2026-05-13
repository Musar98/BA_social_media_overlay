export const CLIP_TO_UV = `#version 300 es
/*
Takes 2D positions (a_pos) in clip space (range -1 to 1)
Converts those positions into UV coordinates (0 → 1)
Flips the Y-axis so textures don't appear upside down
Passes the UVs to the fragment shader
Outputs the vertex position unchanged to the screen
*/

// Input vertex position in clip space (-1 to +1)
in vec2 a_pos;

// UV coordinates passed to fragment shader (0 to 1)
out vec2 v_uv;

void main() {
  // Convert clip space (-1..1) → UV space (0..1)
  // x: map -1 → 0 and +1 → 1
  // y: map -1 → 1 and +1 → 0 (flip Y axis)
  v_uv = vec2(0.5f * (a_pos.x + 1.0f),      // scale and shift X
  1.0f - 0.5f * (a_pos.y + 1.0f) // scale, shift, then flip Y
  );

  // Output final position (no transformation)
  gl_Position = vec4(a_pos, 0.0f, 1.0f);
}`;

/* Inline SVG icon set (24x24, 1.75px stroke). Consistent across platforms — no emoji dependency. */
const ICON_PATHS = {
  grid: "M3 3h8v8H3zM13 3h8v8h-8zM3 13h8v8H3zM13 13h8v8h-8z",
  stethoscope: "M6 4v6a6 6 0 0 0 12 0V4M18 15v2a4 4 0 0 1-8 0M18 15a2 2 0 1 0 0 .01",
  folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  dna: "M4 4c0 6 16 10 16 16M20 4c0 6-16 10-16 16M6 8h12M6 16h12M9 12h6",
  chart: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  atom: "M12 12h.01M12 2c3 0 4 4.5 4 10s-1 10-4 10-4-4.5-4-10 1-10 4-10zM3.3 7c1.5-2.6 6-1.4 10.7 1.3S21.7 14.4 20.7 17s-6 1.4-10.7-1.3S1.8 9.6 3.3 7zM3.3 17c-1.5-2.6 3-6.6 7.7-9.3S19.2 4.4 20.7 7s-3 6.6-7.7 9.3S4.8 19.6 3.3 17z",
  settings: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z",
  file: "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5M9 13h6M9 17h6",
  scan: "M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2M3 12h18",
  search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3",
  userplus: "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM19 8v6M22 11h-6",
  keyboard: "M3 6h18v12H3zM7 10h.01M11 10h.01M15 10h.01M7 14h10",
  heart: "M19.5 12.6 12 20l-7.5-7.4a4.5 4.5 0 1 1 6.4-6.4l1.1 1 1.1-1a4.5 4.5 0 1 1 6.4 6.4z",
  droplet: "M12 2.7 6.3 9.4a7 7 0 1 0 11.4 0z",
  brain: "M9.5 2A2.5 2.5 0 0 0 7 4.5v.6A3 3 0 0 0 4.4 8 3 3 0 0 0 4 12a3 3 0 0 0 1 5.5A2.5 2.5 0 0 0 9.5 22h.5V2zM14.5 2A2.5 2.5 0 0 1 17 4.5v.6a3 3 0 0 1 2.6 2.9 3 3 0 0 1 .4 4 3 3 0 0 1-1 5.5A2.5 2.5 0 0 1 14.5 22H14V2z",
  ribbon: "M12 2a5 5 0 1 0 0 10 5 5 0 0 0 0-10zM8.5 11 5 22l4-2 3 2 3-2 4 2-3.5-11",
  plus: "M12 5v14M5 12h14",
  download: "M12 3v12M6 11l6 6 6-6M4 21h16",
  mail: "M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zM3 7l9 6 9-6",
  check: "M5 12.5 9.5 17 19 7",
  alert: "M12 3 2 21h20zM12 10v4M12 18h.01",
  user: "M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3 2",
  layers: "M12 3 2 8l10 5 10-5zM2 12l10 5 10-5M2 16l10 5 10-5",
  cpu: "M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3M6 6h12v12H6zM9 9h6v6H9z",
  activity: "M2 12h4l3-8 4 16 3-8h6",
  logout: "M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9",
  arrowright: "M5 12h14M13 6l6 6-6 6",
  edit: "M12 20h9M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4z",
  upload: "M12 21V9M6 13l6-6 6 6M4 3h16",
  shield: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
};
const CATEGORY_ICON = { cardiovascular: "heart", diabetes: "droplet", neurological: "brain", cancer: "ribbon", other: "dna" };
function icon(name, size = 20, cls = "") {
  const p = ICON_PATHS[name] || ICON_PATHS.grid;
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24"); s.setAttribute("width", size); s.setAttribute("height", size); s.setAttribute("fill", "none");
  s.setAttribute("stroke", "currentColor"); s.setAttribute("stroke-width", "1.75"); s.setAttribute("stroke-linecap", "round"); s.setAttribute("stroke-linejoin", "round");
  s.setAttribute("class", "svgi " + cls); s.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path"); path.setAttribute("d", p); s.append(path); return s;
}
/* disease icon: SVG by category (falls back to the dataset's own glyph) */
const dIcon = (d, size = 22) => icon(CATEGORY_ICON[d.category] || "dna", size, "cat-" + (d.category || "other"));
/* fill static <span data-icon="..."> slots in the shell */
document.querySelectorAll("[data-icon]").forEach(n => { n.innerHTML = ""; n.append(icon(n.dataset.icon, +(n.dataset.size || 18))); });

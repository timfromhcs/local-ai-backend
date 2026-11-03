/* LocalAI Gateway service worker.
   Precache the shell so the page opens offline; models are NOT cached here (they are large and
   the browser handles those caches). Bump SW_VERSION to invalidate. */
const SW_VERSION = "v2";

// A cached redirect may not be returned for a navigation (Vercel's cleanUrls turns
// /index.html into a 308). Re-wrap the body in a plain Response so the browser accepts it.
async function shellResponse() {
  const cached = (await caches.match("/")) || (await caches.match("/index.html"));
  if (!cached) return Response.error();
  return new Response(cached.body, {
    status: 200,
    statusText: "OK",
    headers: cached.headers,
  });
}
const SHELL = [
  "/",
  "/styles.css",
  "/i18n.js",
  "/app.js",
  "/manifest.webmanifest",
  "/assets/logo.svg",
  "/assets/hero.svg",
  "/assets/diagram.svg",
  "/assets/icon-192.png",
  "/assets/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SW_VERSION)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
      .catch(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== SW_VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  // never intercept the local gateway or model downloads
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/v1/") || url.pathname === "/healthz") return;

  // navigations: network first, fall back to the cached shell so the app opens offline
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(SW_VERSION).then((cache) => cache.put("/", copy)).catch(() => {});
          return response;
        })
        .catch(() => shellResponse()),
    );
    return;
  }

  // assets: stale-while-revalidate
  event.respondWith(
    caches.match(request).then((hit) => {
      const network = fetch(request)
        .then((response) => {
          if (response.ok) {
            const copy = response.clone();
            caches.open(SW_VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
          }
          return response;
        })
        .catch(() => hit);
      return hit || network;
    }),
  );
});

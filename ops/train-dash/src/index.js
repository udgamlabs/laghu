// Udgam Labs private training dashboard.
//   GET  /<VIEW_TOKEN>/                 dashboard page (unguessable link = the only way in)
//   GET  /<VIEW_TOKEN>/api/runs         runs with their last update time
//   GET  /<VIEW_TOKEN>/api/run/<name>   one run's latest snapshot (JSON pushed by the Mac)
//   GET  /<VIEW_TOKEN>/public/          PREVIEW of the public page (sanitized "<run>.public" records only)
//   GET  /live/, /live/api/run/<run>    the PUBLIC page: OFF unless env PUBLIC_ENABLED == "1" (needs the user's go),
//                                       and even then it can only read records whose name ends in ".public"
//   POST /api/push                      Authorization: Bearer <PUSH_SECRET>; body = snapshot JSON with "run"
// Everything else is a plain 404, so the existence of the dashboard is not revealed.
import PAGE from "./page.html";
import PUBLIC_PAGE from "./public.html";

const enc = new TextEncoder();
function safeEqual(a, b) {
  const x = enc.encode(String(a || "")), y = enc.encode(String(b || ""));
  if (x.length !== y.length || x.length === 0) return false;
  return crypto.subtle.timingSafeEqual(x, y);
}
const HDRS = { "Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow", "Referrer-Policy": "no-referrer",
               "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY" };
const notFound = () => new Response("Not found", { status: 404, headers: HDRS });
const json = (o, status = 200) => new Response(JSON.stringify(o), { status, headers: { ...HDRS, "Content-Type": "application/json" } });

async function readRun(env, name) {
  const row = await env.DB.prepare("SELECT updated_at, json FROM snapshots WHERE run = ?1").bind(name).first();
  if (!row) return json({ error: "no such run" }, 404);
  return new Response(`{"updated_at":${row.updated_at},"data":${row.json}}`, { headers: { ...HDRS, "Content-Type": "application/json" } });
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length === 2 && parts[0] === "api" && parts[1] === "push") {
      if (req.method !== "POST") return notFound();
      const auth = (req.headers.get("authorization") || "").replace(/^Bearer /, "");
      if (!safeEqual(auth, env.PUSH_SECRET)) return notFound();
      const body = await req.text();
      if (body.length > 1_900_000) return json({ error: "too large" }, 413);
      let d; try { d = JSON.parse(body); } catch { return json({ error: "bad json" }, 400); }
      const run = String(d.run || "");
      if (!/^[A-Za-z0-9._-]{1,100}$/.test(run)) return json({ error: "bad run name" }, 400);
      await env.DB.prepare("INSERT INTO snapshots (run, updated_at, json) VALUES (?1, ?2, ?3) " +
        "ON CONFLICT(run) DO UPDATE SET updated_at = excluded.updated_at, json = excluded.json").bind(run, Date.now(), body).run();
      return json({ ok: true });
    }
    if (parts[0] === "live") {
      if (env.PUBLIC_ENABLED !== "1" || req.method !== "GET") return notFound();
      if (parts.length === 1) {
        if (!url.pathname.endsWith("/")) return Response.redirect(url.origin + "/live/", 302);
        return new Response(PUBLIC_PAGE.replace("__API__", "/live/api/"), { headers: { ...HDRS, "Content-Type": "text/html; charset=utf-8" } });
      }
      if (parts[1] === "api" && parts[2] === "run" && parts.length === 4 && parts[3].endsWith(".public")) return readRun(env, parts[3]);
      return notFound();
    }
    if (parts.length === 0 || !safeEqual(parts[0], env.VIEW_TOKEN) || req.method !== "GET") return notFound();
    if (parts[1] === "public" && parts.length === 2) {
      if (!url.pathname.endsWith("/")) return Response.redirect(url.origin + url.pathname + "/", 302);
      return new Response(PUBLIC_PAGE.replace("__API__", "/" + parts[0] + "/api/"), { headers: { ...HDRS, "Content-Type": "text/html; charset=utf-8" } });
    }
    if (parts.length === 1) {
      if (!url.pathname.endsWith("/")) return Response.redirect(url.origin + url.pathname + "/", 302);
      return new Response(PAGE, { headers: { ...HDRS, "Content-Type": "text/html; charset=utf-8" } });
    }
    if (parts[1] === "api" && parts[2] === "runs" && parts.length === 3) {
      const { results } = await env.DB.prepare("SELECT run, updated_at FROM snapshots WHERE run NOT LIKE '%.public' ORDER BY updated_at DESC LIMIT 50").all();
      return json({ runs: results });
    }
    if (parts[1] === "api" && parts[2] === "run" && parts.length === 4) return readRun(env, parts[3]);
    return notFound();
  },
};

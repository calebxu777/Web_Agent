# Frontend

This is the Next.js frontend for the Commerce Agent MVP.

## Local Development

From [`frontend/`](./):

```bash
npm install
npm run dev
```

The app runs on `http://localhost:3000`.

## Backend Connection

The frontend proxies requests through Next.js API routes and expects:

```env
BACKEND_URL=http://127.0.0.1:8011
```

For local development, point `BACKEND_URL` at your local FastAPI backend.

## Vercel Deployment

Deploy this folder as a separate Vercel project with:

- Framework Preset: `Next.js`
- Root Directory: `frontend`
- Environment Variable: `BACKEND_URL`

Example:

```env
BACKEND_URL=http://YOUR_VM_PUBLIC_IP:8011
```

For the current MVP, the frontend talks to the GCP VM backend over the public backend URL. Later, this should move behind HTTPS on a domain rather than raw port `8011`.

## API Routes

The frontend includes proxy routes for:

- `/api/chat`
- `/api/nickname`
- `/api/feedback`

These routes forward requests to the FastAPI backend defined by `BACKEND_URL`.

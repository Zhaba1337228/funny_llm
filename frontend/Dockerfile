FROM node:22-alpine AS builder

WORKDIR /app

COPY frontend/package.json /app/package.json
COPY frontend/package-lock.json /app/package-lock.json
RUN npm ci

COPY frontend /app
RUN npm run build

FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

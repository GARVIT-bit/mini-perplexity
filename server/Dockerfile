# Express backend.
FROM node:24-alpine

ENV NODE_ENV=production
WORKDIR /app

# Dependencies first: this layer is reused until package*.json changes
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY src/ src/

USER node
EXPOSE 3001
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:3001/api/history?limit=1 || exit 1
CMD ["node", "src/index.js"]

FROM node:latest

RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json .

RUN npm i 

COPY . .

EXPOSE 7860

CMD ["node", "--max-old-space-size=8192", "index.js"]
FROM node:23-alpine
CMD ["echo", "Building Docker image for bot"]

COPY package*.json ./

RUN cat /etc/resolv.conf

RUN npm install

COPY . .

EXPOSE 3000

CMD ["node", "."]
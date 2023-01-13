FROM node

WORKDIR /app


RUN npm install

COPY . .

EXPOSE 80


CMD ["npm", "start"]
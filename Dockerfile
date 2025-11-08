FROM mysql:8.0

ENV MYSQL_ROOT_PASSWORD=root
ENV MYSQL_DATABASE=login_app

COPY backup.sql /docker-entrypoint-initdb.d/

EXPOSE 3306

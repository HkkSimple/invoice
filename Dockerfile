FROM kksimple/invoice:env

ADD invoice invoice

WORKDIR invoice

EXPOSE 30500    

ENTRYPOINT ["/bin/bash", "run.sh"]
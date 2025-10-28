
Spark is a powerful distributed computing framework widely used for big data processing and analytics.

``` tree
docker/
    config/
        log4j.properties
        spark-defaults.conf
    data/
    scripts/
    logs/
    compose.yaml
    .gitignore
```

=== "Docker Directory Structure"

    - `config/`: Config files to customize Spark settings.
        - `log4j.properties`: Spark logging configuration.
        - `spark-defaults.conf`: Standard Spark configuration settings.
    - `data/`: Directory to store datasets used in examples.
    - `scripts/`: Helper scripts to start and manage the Spark environment.
    - `logs/`: Directory where Spark logs will be stored.
    - `compose.yaml`: Docker Compose file to configure and start Spark services.
    - `.gitignore`: File to ignore unnecessary files and directories in version control.

=== "Setup Spark Cluster with Docker Compose"

    To set up a Spark cluster using Docker Compose, use the following configuration ([compose.yaml](./docker/compose.yaml){:download="compose.yaml"}):

    ``` { .yaml .copy .select linenums="1" }
    --8<-- "docs/classes/pyspark/docker/compose.yaml"
    ```

    This configuration defines a Spark master, three worker nodes, and a history server, along with necessary environment variables and volume mounts for configuration files and data.

    ```mermaid
    graph LR
        user[User]
        subgraph Spark Cluster
            direction LR
            master[Master Node]
            worker1[Worker Node 1]
            worker2[Worker Node 2]
            worker3[Worker Node 3]
            history[History Server]
        end
        user -->|Submits Jobs| master
        user -->|Accesses UI| history
        master --> worker1
        master --> worker2
        master --> worker3
        master --> history
    ```

    To start the Spark cluster, navigate to the `docker/` directory and run:

    ```bash
    docker compose up -d --build
    ```

[^1]: [Running Spark using Docker Compose](https://karlchris.github.io/data-engineering/projects/spark-docker/#spark-history-server){:target="_blank"}
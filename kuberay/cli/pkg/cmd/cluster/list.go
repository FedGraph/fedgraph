package cluster

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/ray-project/kuberay/cli/pkg/cmdutil"
	"github.com/ray-project/kuberay/proto/go_client"
	"github.com/spf13/cobra"
)

type ListOptions struct {
	namespace string
}

func newCmdList() *cobra.Command {
	opts := ListOptions{}

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List all ray clusters",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return listCluster(opts)
		},
	}

	cmd.Flags().StringVarP(&opts.namespace, "namespace", "n", "",
		"kubernetes namespace where the cluster is provisioned")

	return cmd
}

func listCluster(opts ListOptions) error {
	// Get gRPC connection
	conn, err := cmdutil.GetGrpcConn()
	if err != nil {
		return err
	}
	defer conn.Close()

	// build gRPC client
	client := go_client.NewClusterServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var clusters []*go_client.Cluster
	if len(opts.namespace) == 0 {
		r, err := client.ListAllClusters(ctx, &go_client.ListAllClustersRequest{})
		if err != nil {
			log.Fatalf("could not list all clusters %v", err)
		}
		clusters = r.GetClusters()
	} else {
		r, err := client.ListCluster(ctx, &go_client.ListClustersRequest{
			Namespace: opts.namespace,
		})
		if err != nil {
			log.Fatalf("could not list clusters %v", err)
		}
		clusters = r.GetClusters()
	}
	rows, maxNumberWorkerGroups := convertClustersToStrings(clusters)
	header := []string{"Name", "Namespace", "User", "Version", "Environment", "Created At", "Head Image", "Head Compute Template", "Head Service Type"}
	for i := 0; i < maxNumberWorkerGroups; i++ {
		header = append(header, "Worker Group Name", "Worker Image", "Worker ComputeTemplate")
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader(header)
	table.AppendBulk(rows)
	table.Render()

	return nil
}

func convertClustersToStrings(clusters []*go_client.Cluster) ([][]string, int) {
	var data [][]string

	// max number of worker groups among all clusters. This will decide how wide the table is.
	maxNumberWorkerGroups := 0
	for _, r := range clusters {
		row, nWorkerGroups := convertClusterToString(r)
		data = append(data, row)

		if nWorkerGroups > maxNumberWorkerGroups {
			maxNumberWorkerGroups = nWorkerGroups
		}
	}

	return data, maxNumberWorkerGroups
}

func convertClusterToString(r *go_client.Cluster) ([]string, int) {
	headResource := r.GetClusterSpec().GetHeadGroupSpec()
	workerGroups := r.GetClusterSpec().GetWorkerGroupSpec()
	line := []string{
		r.GetName(), r.GetNamespace(), r.GetUser(), r.GetVersion(), r.GetEnvironment().String(), r.GetCreatedAt().AsTime().String(),
		headResource.GetImage(), headResource.GetComputeTemplate(), headResource.GetServiceType(),
	}
	nWorkGroups := len(workerGroups)

	for _, workerGroup := range workerGroups {
		line = append(line, workerGroup.GetGroupName(), workerGroup.GetImage(), workerGroup.GetComputeTemplate())
	}
	return line, nWorkGroups
}
